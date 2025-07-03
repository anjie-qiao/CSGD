import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import os

from models.transformer import Denoiser
from diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalTransition

from diffusion import diffusion_utils
from metrics.train_loss import TrainLossScore
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
import utils

import diffusion.graph_lib as graph_lib
import diffusion.noise_lib as noise_lib
from torch.optim.lr_scheduler import LambdaLR
import math
import random

class Graph_DiT(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools):
        super().__init__()
        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.test_only = cfg.general.test_only
        self.guidance_target = getattr(cfg.dataset, 'guidance_target', None)

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist
        active_index = dataset_infos.active_index

        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps
        self.guide_scale = cfg.model.guide_scale
        self.graph_type = cfg.graph.type

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        if self.graph_type == "absorb":
            self.Xdim_output = output_dims['X'] + 1
            self.Edim_output = output_dims['E'] + 1
        else:
            self.Xdim_output = output_dims['X']
            self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist
        self.active_index = active_index
        self.dataset_info = dataset_infos

        self.train_loss = TrainLossScore(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_collection = []

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_collection = []

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.max_n_nodes = dataset_infos.max_n_nodes

        self.model = Denoiser(max_n_nodes=self.max_n_nodes,
                        hidden_size=cfg.model.hidden_size,
                        depth=cfg.model.depth,
                        num_heads=cfg.model.num_heads,
                        mlp_ratio=cfg.model.mlp_ratio,
                        drop_condition=cfg.model.drop_condition,
                        Xdim=(self.Xdim + 1 if self.graph_type == "absorb" else self.Xdim),
                        Edim=(self.Edim + 1 if self.graph_type == "absorb" else self.Edim),
                        ydim=self.ydim,
                        task_type=dataset_infos.task_type,
                        graph_type=self.graph_type,)

        self.noise = noise_lib.get_noise(cfg)

        x_marginals = self.dataset_info.node_types.float() / torch.sum(self.dataset_info.node_types.float())
        
        e_marginals = self.dataset_info.edge_types.float() / torch.sum(self.dataset_info.edge_types.float())
        x_marginals = x_marginals / (x_marginals ).sum()
        e_marginals = e_marginals / (e_marginals ).sum()

        xe_conditions = self.dataset_info.transition_E.float()
        xe_conditions = xe_conditions[self.active_index][:, self.active_index] 
        
        xe_conditions = xe_conditions.sum(dim=1) 
        ex_conditions = xe_conditions.t()
        xe_conditions = xe_conditions / xe_conditions.sum(dim=-1, keepdim=True)
        ex_conditions = ex_conditions / ex_conditions.sum(dim=-1, keepdim=True)
        
        self.transition_model = MarginalTransition(x_marginals=x_marginals, 
                                                          e_marginals=e_marginals, 
                                                          xe_conditions=xe_conditions,
                                                          ex_conditions=ex_conditions,
                                                          y_classes=self.ydim_output,
                                                          n_nodes=self.max_n_nodes)

        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals, y=None)

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps

        self.best_val_nll = 1e8
        self.val_counter = 0
        self.batch_size = self.cfg.train.batch_size

        self.sampling_eps = cfg.noise.sampling_eps
        cfg.graph.x_dim = output_dims['X']
        cfg.graph.e_dim = output_dims['E']
        self.graph = graph_lib.get_graph(cfg)
        self.warmup_epochs = cfg.train.warmup_epochs
        self.beta1 = cfg.train.beta1
        self.beta2 = cfg.train.beta2
        self.train_eps = cfg.train.eps
        self.n_epochs = cfg.train.n_epochs
        self.min_lr_factor = cfg.train.min_lr_factor
        self.start_decay  = int(self.n_epochs * 0.6)

    def forward(self, noisy_data, condition_index, unconditioned=False, train=False):
        x, e, y = noisy_data['X_t'].float(), noisy_data['E_t'].float(), noisy_data['y_t'].float().clone()
        node_mask, t, sigma =  noisy_data['node_mask'], noisy_data['t'], noisy_data['sigma']
        index_x, index_e = noisy_data['index_x'], noisy_data['index_e']
        # in contionous-time diffusion, sigma replaces the timestep t as the condition
        pred = self.model(x, e, node_mask, y=y, t=sigma, index_x=index_x, index_e=index_e, unconditioned=unconditioned, condition_index=condition_index, train=train)
        return pred
        
    def training_step(self, data, i):
        # 118 is the number  in the periodic table (H to Og).
        data_x = F.one_hot(data.x, num_classes=118).float()[:, self.active_index]
        data_edge_attr = F.one_hot(data.edge_attr, num_classes=5).float()

        dense_data, node_mask = utils.to_dense(data_x, data.edge_index, data_edge_attr, data.batch, self.max_n_nodes)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        #data.y = [SA + SC + target properties (such as O2-N2-CO2)]
        noisy_data = self.apply_noise(X, E, data.y, node_mask)

        # randomly select single condition
        # polymers (SA/SC, O2, N2, CO2)  ydim = 5 
        # BACE/BBBP/HIV (SA/SC， Class)  ydim = 3
        #selected_cond_idx = random.randint(0, self.ydim - 3) 
        # probs = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        # probs = probs / probs.sum()
        # # (bs, 1)
        # selected_cond_idx = torch.multinomial(probs, num_samples=X.shape[0], replacement=True).unsqueeze(1)  
        selected_cond_idx = torch.argsort(torch.rand(X.shape[0], self.ydim - 1, device=self.device), dim=1)
     
        pred = self.forward(noisy_data, condition_index=selected_cond_idx, train=True)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                            true_X=X, true_E=E, true_y=data.y, node_mask=node_mask, 
                            noisy_data=noisy_data,
                            sigma = noisy_data['sigma'],
                            graph = self.graph,
                            log=i % self.log_every_steps == 0)
           
        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                        log=i % self.log_every_steps == 0)
        self.log(f'loss', loss, batch_size=X.size(0), sync_dist=True)
        return {'loss': loss}


    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

        def lr_lambda(current_epoch: int):
            if current_epoch < self.warmup_epochs:
                return float(current_epoch) / float(max(1, self.warmup_epochs))
            if current_epoch < self.start_decay:
                return 1.0
            
            progress = float(current_epoch - self.start_decay) / float(max(1, (self.n_epochs - self.start_decay)))
            # cosine decay
            return  self.min_lr_factor + (1 - self.min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'epoch',  
            'frequency': 1,
            'name': 'learning_rate'
        }
        return [optimizer], [scheduler]
    
    def on_fit_start(self) -> None:
        self.train_iterations = self.trainer.datamodule.training_iterations
        print('on fit train iteration:', self.train_iterations)
        print("Size of the input features Xdim {}, Edim {}, ydim {}".format(self.Xdim, self.Edim, self.ydim))

    def on_train_epoch_start(self) -> None:
        if self.current_epoch / self.trainer.max_epochs in [0.25, 0.5, 0.75, 1.0]:
            print("Starting train epoch {}/{}...".format(self.current_epoch, self.trainer.max_epochs))
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        if self.current_epoch / self.trainer.max_epochs in [0.25, 0.5, 0.75, 1.0]:
            log = True
        else:
            log = False
        self.train_loss.log_epoch_metrics(self.current_epoch, self.start_epoch_time, log)
        self.train_metrics.log_epoch_metrics(self.current_epoch, log)

    def on_validation_epoch_start(self) -> None:
        # self.val_nll.reset()
        # self.val_X_kl.reset()
        # self.val_E_kl.reset()
        # self.val_X_logp.reset()
        # self.val_E_logp.reset()
        self.sampling_metrics.reset()
        self.val_y_collection = []

    @torch.no_grad()
    def validation_step(self, data, i):
        data_x = F.one_hot(data.x, num_classes=118).float()[:, self.active_index]
        data_edge_attr = F.one_hot(data.edge_attr, num_classes=5).float()
        dense_data, node_mask = utils.to_dense(data_x, data.edge_index, data_edge_attr, data.batch, self.max_n_nodes)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        selected_cond_idx = torch.full((dense_data.X.shape[0], 1), fill_value=0, dtype=torch.long, device=self.device)
        pred = self.forward(noisy_data, condition_index=selected_cond_idx)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False)
        self.val_y_collection.append(data.y)
        self.log(f'valid_nll', nll, batch_size=data.x.size(0), sync_dist=True)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        # metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
        #            self.val_X_logp.compute(), self.val_E_logp.compute()]
        
        # if self.current_epoch / self.trainer.max_epochs in [0.25, 0.5, 0.75, 1.0]:
        #     print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
        #         f"Val Edge type KL: {metrics[2] :.2f}", 'Val loss: %.2f \t Best :  %.2f\n' % (metrics[0], self.best_val_nll))

        # # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        # self.log("val/NLL",  metrics[0], sync_dist=True)

        # if metrics[0] < self.best_val_nll:
        #     self.best_val_nll = metrics[0]

        self.val_counter += 1
        
        if self.val_counter % self.cfg.general.sample_every_val == 0 and self.val_counter > 1:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples, all_ys, ident = [], [], 0

            self.val_y_collection = torch.cat(self.val_y_collection, dim=0)
            num_examples = self.val_y_collection.size(0)
            start_index = 0
            while samples_left_to_generate > 0:                
                bs = int(1 * self.cfg.train.batch_size)
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)

                if start_index + to_generate > num_examples:
                    start_index = 0
                if to_generate > num_examples:
                    ratio = to_generate // num_examples
                    self.val_y_collection = self.val_y_collection.repeat(ratio+1, 1)
                    num_examples = self.val_y_collection.size(0)
                batch_y = self.val_y_collection[start_index:start_index + to_generate]                
                all_ys.append(batch_y)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, y=batch_y,
                                                save_final=to_save,
                                                keep_chain=chains_save,
                                                number_chain_steps=self.number_chain_steps))
                ident += to_generate
                start_index += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save

            print(f"Computing sampling metrics", ' ...')
            valid_smiles = self.sampling_metrics(samples, all_ys, self.name, self.current_epoch, val_counter=-1, test=False)
            print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b0/')
            self.visualization_tools.visualize_by_smiles(result_path, valid_smiles, self.cfg.general.samples_to_save)
            self.sampling_metrics.reset()

    def on_test_epoch_start(self) -> None:
        print("Starting test...")
        # self.test_nll.reset()
        # self.test_X_kl.reset()
        # self.test_E_kl.reset()
        # self.test_X_logp.reset()
        # self.test_E_logp.reset()
        self.test_y_collection = []
    
    @torch.no_grad()
    def test_step(self, data, i):
        data_x = F.one_hot(data.x, num_classes=118).float()[:, self.active_index]
        data_edge_attr = F.one_hot(data.edge_attr, num_classes=5).float()

        dense_data, node_mask = utils.to_dense(data_x, data.edge_index, data_edge_attr, data.batch, self.max_n_nodes)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        selected_cond_idx = torch.full((dense_data.X.shape[0], 1), fill_value=0, dtype=torch.long, device=self.device)
        pred = self.forward(noisy_data, condition_index=selected_cond_idx)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
        self.test_y_collection.append(data.y)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        # metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
        #            self.test_X_logp.compute(), self.test_E_logp.compute()]

        # print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
        #       f"Test Edge type KL: {metrics[2] :.2f}")

        ## final epcoh
        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        #samples_left_to_generate = 1200
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples, all_ys, batch_id = [], [], 0
        test_y_collection = torch.cat(self.test_y_collection, dim=0)
        num_examples = test_y_collection.size(0)
        if self.cfg.general.final_model_samples_to_generate > num_examples:
            ratio = self.cfg.general.final_model_samples_to_generate // num_examples
            test_y_collection = test_y_collection.repeat(ratio+1, 1)
            num_examples = test_y_collection.size(0)
        
        while samples_left_to_generate > 0:
            print(f'samples left to generate: {samples_left_to_generate}/'
                f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = int(1 * self.cfg.train.batch_size)
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            batch_y = test_y_collection[batch_id : batch_id + to_generate]

            cur_sample = self.sample_batch(batch_id, to_generate, batch_y, save_final=to_save,
                                            keep_chain=chains_save, number_chain_steps=self.number_chain_steps)
            samples = samples + cur_sample
            del cur_sample
            torch.cuda.empty_cache()
            
            all_ys.append(batch_y)
            batch_id += to_generate

            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
            
        print(f"final Computing sampling metrics...")
        self.sampling_metrics.reset()
        self.sampling_metrics(samples, all_ys, self.name, self.current_epoch, self.val_counter, test=True)
        self.sampling_metrics.reset()
        print(f"Done.")
            

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)
        
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        
        bs, n, d = X.shape
        X_all = torch.cat([X, E.reshape(bs, n, -1)], dim=-1)
        prob_all = X_all @ Qtb.X
        probX = prob_all[:, :, :self.Xdim_output]
        probE = prob_all[:, :, self.Xdim_output:].reshape((bs, n, n, -1))

        assert probX.shape == X.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        X_all = torch.cat([X, E.reshape(bs, n, -1)], dim=-1).float()
        Xt_all = torch.cat([noisy_data['X_t'], noisy_data['E_t'].reshape(bs, n, -1)], dim=-1).float()
        pred_probs_all = torch.cat([pred_probs_X, pred_probs_E.reshape(bs, n, -1)], dim=-1).float()

        prob_true = diffusion_utils.posterior_distributions(X=X_all, X_t=Xt_all, Qt=Qt, Qsb=Qsb, Qtb=Qtb, X_dim=self.Xdim_output)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_all, X_t=Xt_all, Qt=Qt, Qsb=Qsb, Qtb=Qtb, X_dim=self.Xdim_output)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))

        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, y, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_0, self.device)
        
        bs, n, d = X.shape
        X_all = torch.cat([X, E.reshape(bs, n, -1)], dim=-1)
        prob_all = X_all @ Q0.X
        probX0 = prob_all[:, :, :self.Xdim_output]
        probE0 = prob_all[:, :, self.Xdim_output:].reshape((bs, n, n, -1))

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()

        assert (X.shape == X0.shape) and (E.shape == E0.shape)
        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y)}
        pred0 = self.forward(noisy_data)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = None

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """
        # continuous time step sampling
        bs, n, _ = X.shape
        t = (1 - self.sampling_eps) * torch.rand(bs, device=X.device) + self.sampling_eps
        # (bs)
        sigma, dsigma = self.noise(t)
        #(bs, N)
        i_x = X.argmax(dim=-1) 
        #(bs, N)
        i_x_pert = self.graph.sample_transition(i_x, sigma[:, None])
        #(bs, N, d)
        X_t = F.one_hot(i_x_pert, num_classes=self.Xdim_output)

        #(bs, N, N)
        i_e = E.argmax(dim=-1)
        #(bs, N, N)
        i_e_pert = self.graph.sample_transition(i_e,  sigma[:, None, None])
        #(bs, N, N, d)
        E_t_upper = torch.triu(i_e_pert, diagonal=1)
        E_t = (E_t_upper + E_t_upper.transpose(1, 2))
    
        E_t = F.one_hot(E_t, num_classes=self.Edim_output)

        y_t = y
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y_t).type_as(X_t).mask(node_mask)

        noisy_data = {'t': t,  'sigma': sigma, 'dsigma': dsigma, 'index_x': i_x_pert, 'index_e': E_t.argmax(dim=-1), 
        'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        # t = noisy_data['t']

        # # 1.
        # N = node_mask.sum(1).long()
        # log_pN = self.node_dist.log_prob(N)

        # # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        # kl_prior = self.kl_prior(X, E, node_mask)

        # # 3. Diffusion loss
        # loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # # 4. Reconstruction loss
        # # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        # prob0 = self.reconstruction_logp(t, X, E, y, node_mask)

        # eps = 1e-8
        # loss_term_0 = self.val_X_logp(X * (prob0.X+eps).log()) + self.val_E_logp(E * (prob0.E+eps).log())

        # # Combine terms
        # nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        # assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # # Update NLL metric object and return batch nll
        # nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch
        
        return 0.0
    
    @torch.no_grad()
    def sample_batch(self, batch_id, batch_size, y, keep_chain, number_chain_steps, save_final, num_nodes=None
     , eps=1e-5):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file (disabled)
        :param keep_chain_steps: number of timesteps to save for each chain (disabled)
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = self.max_n_nodes
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
     
        X = self.graph.sample_limit((batch_size, n_max),  sample_type='node').to(self.device)   
        X = F.one_hot(X, num_classes=self.Xdim_output)
        E = self.graph.sample_limit((batch_size, n_max , n_max),  sample_type='edge').to(self.device)
        E_t_upper = torch.triu(E, diagonal=1)
        E_t = (E_t_upper + E_t_upper.transpose(1, 2))
        E_t = F.one_hot(E_t, num_classes=self.Edim_output)

        Z_t = utils.PlaceHolder(X=X, E=E_t, y=y).type_as(X).mask(node_mask)
        X = Z_t.X
        E = Z_t.E

        timesteps = torch.linspace(1, eps, self.T + 1, device=self.device)
        dt = (1 - eps) / self.T
        for i in range(self.T):
            t = timesteps[i] * torch.ones(X.shape[0], 1, device=self.device)
            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(t, dt, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        
        #follow the final denosing process of SEDD
        t = timesteps[-1] * torch.ones(X.shape[0], 1, device=self.device)
        sampled_s, discrete_sampled_s = self.sample_p_z0(t, dt, X, E, y, node_mask)

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        
        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
        print("sampling completed!")
        return molecule_list

    def sample_p_zs_given_zt(self, t, dt, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
           
        bs, n, dxs = X_t.shape
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - dt)[0]
        dsigma = curr_sigma - next_sigma

        noisy_data = {'t': t,  'sigma': curr_sigma.squeeze(-1), 'dsigma': dsigma.squeeze(-1), 'X_t': X_t, 'E_t': E_t, 'y_t': y_t
        ,'index_x': X_t.argmax(dim=-1) ,'index_e': E_t.argmax(dim=-1) ,'node_mask': node_mask}
       
        def get_prob(noisy_data, unconditioned=False, condition_index=0):
            pred = self.forward(noisy_data, unconditioned=unconditioned, condition_index=condition_index)
            return pred.X, pred.E

        ## Guidance
        if self.guidance_target is not None and self.guide_scale is not None and self.guide_scale != 1:
            cond_idx = torch.full((bs, 1), fill_value=0, dtype=torch.long, device=self.device)
            base_logscore_X, base_logscore_E = get_prob(noisy_data, unconditioned=True, condition_index=cond_idx)
            guided_logscore_X = torch.zeros(X_t.shape, device=self.device, dtype=torch.float32)
            guided_logscore_E = torch.zeros(E_t.shape, device=self.device, dtype=torch.float32)
            #avg_guide_scale =  self.guide_scale / (self.ydim-1)
            #weights = torch.tensor([0.4, 0.2, 0.2, 0.2], device=self.device, dtype=torch.float32)
            weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=self.device, dtype=torch.float32)
            for i in range(self.ydim-1):
                selected_cond_idx = torch.full((bs, 1), fill_value=i, dtype=torch.long, device=self.device)
                logscore_X_cond, logscore_E_cond = get_prob(noisy_data, unconditioned=False, condition_index=selected_cond_idx)
                guided_logscore_X += self.guide_scale * weights[i] * (logscore_X_cond - base_logscore_X)
                guided_logscore_E += self.guide_scale * weights[i] * (logscore_E_cond - base_logscore_E)
            
            guide_X  = base_logscore_X + guided_logscore_X
            guide_E = base_logscore_E + guided_logscore_E

            guide_X = guide_X.exp()
            stag_score_x = self.graph.staggered_score(guide_X, dsigma)
            prob_X = stag_score_x * self.graph.transp_transition(noisy_data['index_x'], dsigma[..., None])
        
            guide_E= guide_E.exp()
            stag_score_e = self.graph.staggered_score(guide_E, dsigma)
            prob_E = stag_score_e * self.graph.transp_transition(noisy_data['index_e'], dsigma[..., None, None])     

        sampled_s = diffusion_utils.gumbel_sample_discrete_features(prob_X, prob_E)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)


    def sample_p_z0(self, t, dt, X_t, E_t, y_t, node_mask):
    
        bs, n, dxs = X_t.shape
        sigma = self.noise(t)[0]
 
        noisy_data = {'t': t,  'sigma': sigma.squeeze(-1), 'X_t': X_t, 'E_t': E_t, 'y_t': y_t
        ,'index_x': X_t.argmax(dim=-1) ,'index_e': E_t.argmax(dim=-1) ,'node_mask': node_mask}
       

        def get_prob(noisy_data, unconditioned=False, condition_index=0):
            pred = self.forward(noisy_data, unconditioned=unconditioned, condition_index=condition_index)
            return pred.X, pred.E

        ## Guidance
        if self.guidance_target is not None and self.guide_scale is not None and self.guide_scale != 1:
            cond_idx = torch.full((bs, 1), fill_value=0, dtype=torch.long, device=self.device)
            base_logscore_X, base_logscore_E = get_prob(noisy_data, unconditioned=True, condition_index=cond_idx)
            guided_logscore_X = torch.zeros(X_t.shape, device=self.device, dtype=torch.float32)
            guided_logscore_E = torch.zeros(E_t.shape, device=self.device, dtype=torch.float32)
            #avg_guide_scale =  self.guide_scale / (self.ydim-1)
            #weights = torch.tensor([0.4, 0.2, 0.2, 0.2], device=self.device, dtype=torch.float32)
            weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=self.device, dtype=torch.float32)
            for i in range(self.ydim-1):
                selected_cond_idx = torch.full((bs, 1), fill_value=i, dtype=torch.long, device=self.device)
                logscore_X_cond, logscore_E_cond = get_prob(noisy_data, unconditioned=False, condition_index=selected_cond_idx)
                guided_logscore_X += self.guide_scale * weights[i] * (logscore_X_cond - base_logscore_X)
                guided_logscore_E += self.guide_scale * weights[i] * (logscore_E_cond - base_logscore_E)
            guide_X  = base_logscore_X + guided_logscore_X
            guide_E = base_logscore_E + guided_logscore_E

            guide_X = guide_X.exp()
            stag_score_x = self.graph.staggered_score(guide_X, sigma)
            prob_X = stag_score_x * self.graph.transp_transition(noisy_data['index_x'], sigma[..., None])

            guide_E= guide_E.exp()
            stag_score_e = self.graph.staggered_score(guide_E, sigma)
            prob_E = stag_score_e * self.graph.transp_transition(noisy_data['index_e'], sigma[..., None, None])      

        if self.graph_type == "absorb":
            prob_X = prob_X[..., :-1]
            prob_E = prob_E[..., :-1]

        sampled_s = diffusion_utils.gumbel_sample_discrete_features(prob_X, prob_E)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)
