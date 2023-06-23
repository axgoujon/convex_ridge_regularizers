import numpy as np
import pandas as pd
from datetime import datetime

class ValidateCoarseToFine():
    """Class to perform a coarse-to-fine hyperparameter tuning"""
    def __init__(self, score, dir_name, exp_name, p1_init=1, p2_init=10, gamma1=3, gamma2=3, decay_factor=0.5, grid_size=3, freeze_p2=False, verbose=True, gamma_stop=1.01, p1_max=float('inf'), p2_max=float('inf'), **kwargs):
        """ Parameters
        ----------
        score: function
            function that takes as input the parameters and return a tuple (psnr, ssim, niter) corresponding to the average metrics over the validation set of a method. Nb: if freeze_p2 is True, the function should take only one parameter
        dir_name: string
            directory where to store the results
        exp_name: string
            name of the experiment to store the results
        p1_init: float
            initial value of the first parameter
        p2_init: float
            initial value of the second parameter
        gamma1: float
            initial multiplicative grid scale for the first parameter
        gamma2: float
            initial multiplicative grid scale for the second parameter
        decay_factor: float
            multiplicative factor to update the grid scale when going finer
        grid_size: int
            size of the local grid explore the performace of the method
        freeze_p2: bool
            if True, the second parameter is not optimized => faster procedure
        verbose: bool

        gamma_stop: float
            if gamma1 < gamma_stop and gamma2 < gamma_stop, the procedure stops. Should be > 1
        gamma_1_stop: float
            if defined, the stopping criterion for the first parameter, otherwise gamma_stop is used
        gamma_2_stop: float
            if defined, the stopping criterion for the first parameter, otherwise gamma_stop is used
        p1_max: float
            upper bound for the first parameter
        p2_max: float
            upper bound for the second parameter
        """
        

        # the score function takes as input the parameters p1 (and possibily p2) and return tuple (psnr, ssim, niter)
        def score_modified(p1, p2):
            if self.freeze_p2:
                return(score(p1))
            else:
                return(score(p1, p2))

        self.score = score_modified

        # dir to store continuously the results
        self.dir_name = dir_name
        # name of the experiment
        self.exp_name = f"validation_scores_{exp_name}"

        # upper-bound, if any
        self.p1_max = p1_max
        self.p2_max = p2_max

        print(p2_max)

        # initiliaze the parameters
        self.p1 = p1_init
        self.p2 = p2_init

        # initialize the grid scale in the two directions
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        # decay factor to update grid scale
        self.decay_factor = decay_factor

        # local grid size
        self.grid_size = grid_size
        # tensor to store temporarily the scores on the local (grid_size x grid_size) grid
        self.psnr_grid = np.zeros((grid_size, grid_size))
        self.ssim_grid = np.zeros((grid_size, grid_size))
 
        # Dataframe to store all results computed
        cols = ["exp", "p1", "p2", "psnr", "ssim", "niter", "date"]
        dtype = ["string", "float", "float", "float", "float", "float", "string"]
        self.scores= pd.concat([pd.Series(name=col, dtype=dt) for col, dt in zip(cols, dtype)], axis=1)


        # verbose
        self.verbose = verbose

        # only optimize over p1
        self.freeze_p2 = freeze_p2

        gamma_1_stop = kwargs.get('gamma_1_stop', None)
        gamma_2_stop = kwargs.get('gamma_2_stop', None)

        if gamma_1_stop is None:
            self.gamma_1_stop = gamma_stop
        else:
            self.gamma_1_stop = gamma_1_stop

        if gamma_2_stop is None:
            self.gamma_2_stop = gamma_stop
        else:
            self.gamma_2_stop = gamma_2_stop

        


    # update scores on the local grid
    def update_scores(self):
        print("---- computing val scores on the local grid ----")
        self.psnr_grid[:, :] = - np.inf
        self.ssim_grid[:, :] = - np.inf

        if self.freeze_p2:
            loop2 = 1
        else:
            loop2 = self.grid_size

        for k in range(self.grid_size):
            for j in range(loop2):


                p1 = self.p1*(self.gamma1)**k
                p2 = self.p2*(self.gamma2)**j

                if self.freeze_p2:
                    print(f"\t --> p1={p1:.2e}")
                else:
                    print(f"\t --> p1={p1:.2e}, p2={p2:.2e}")
                
                # check if psnr already computed for (almost) the same parameters
                df_t = self.scores[((1 - self.scores["p1"]/p1).abs()<(1e-4))]
                df_t = df_t[(df_t["p2"]/p2 - 1).abs()<(1e-4)]
        
                sk = False
                sk_p1 = False
                sk_p2 = False
                if df_t.shape[0] > 0:
                    psnr_t = df_t.iloc[0, 3]
                    ssim_t = df_t.iloc[0, 4]
                    niter_t = df_t.iloc[0, 5]
                    sk = True
                    self.psnr_grid[k, j] = psnr_t
                    self.ssim_grid[k, j] = ssim_t

                elif (p1 > self.p1_max):
                    sk_p1 = True  
                elif (p2 > self.p2_max):
                    sk_p2 = True
                else:
                    psnr_t, ssim_t,  niter_t = self.score(p1, p2)
                    
                    self.psnr_grid[k, j] = psnr_t
                    self.ssim_grid[k, j] = ssim_t

                    df2 = pd.DataFrame([[self.exp_name, p1, p2, psnr_t, ssim_t, niter_t, get_time_str()]], columns=self.scores.columns)
                    self.scores = pd.concat((self.scores, df2))
                    self.save_scores()
                    
        
                if self.verbose:
                    print(20*"=")
                    print(f'\n\n=== Effective iter {len(self.scores)} === Job {self.exp_name.replace("validation_scores_", "")} ===')
                    if not(sk_p1 or sk_p2):
                        print(f"\n \t psnr={psnr_t:.2f}dB (best {self.scores['psnr'].max():.2f}dB)")
                    if sk:
                        print(f" \t \t (computation skipped since found values for p1={df_t.iloc[0, 1]:.3f}, p2={df_t.iloc[0, 2]:.3f})")
                    if sk_p1:
                        print(f" \t \t (skipping value since p1 greater than {self.p1_max})")
                    if sk_p2:
                        print(f" \t \t (skipping value since p2 greater than {self.p2_max})")
                    #print(f"SSIM {ssim_t:.3f} __ running max {self.scores['ssim'].max():.3f}")
    
    def save_scores(self):
        self.scores.to_csv(f"{self.dir_name}/{self.exp_name}.csv")

    # update grid properties based on the scores
    def update_grid(self):
        print(f"\n ----  updating grid ----")
        ind = np.unravel_index(np.argmax(self.psnr_grid, axis=None), self.psnr_grid.shape)

        if ind[0] == 0:
            self.p1 = self.p1 * (self.gamma1)**(-(self.grid_size//2))
            print("-[p1] lower border hit for => shifting grid down")
        elif ind[0] == self.grid_size - 1:
            
            self.p1 = self.p1 * (self.gamma1)**(self.grid_size//2)
            print("-[p1] upper border hit for => shifting grid up")
        else:
            # find new center
            p1_new_center = self.p1*(self.gamma1)**ind[0]
            # reduce scale
            self.gamma1 = (self.gamma1**self.decay_factor)
            # update left point
            self.p1 = p1_new_center*self.gamma1**(-(self.grid_size//2))
            print(f"-[p1] refinining the grid (corner {self.p1:.3e}, scale {self.gamma1:.3e})")
        self.p1 = min(self.p1, self.p1_max)


        if not self.freeze_p2:
            if ind[1] == 0:
                print("-[p2] lower border hit => shifting grid down")
                self.p2 = self.p2 * (self.gamma2)**(-(self.grid_size//2))
                
            elif ind[1] == self.grid_size - 1:
                print("-[p2] upper border hit => shifting grid up")
                self.p2 = self.p2 * (self.gamma2)**(self.grid_size//2)
                
            else:
                # find new center
                p2_new_center = self.p2*(self.gamma2)**ind[0]
                # reduce scale
                self.gamma2 = (self.gamma2**self.decay_factor)
                # update left point
                self.p2 = p2_new_center*self.gamma2**(-(self.grid_size//2))
                print(f"-[p2] refinining the grid (corner {self.p2:.3f}, scale {self.gamma2:.3f})")
            self.p2 = min(self.p2, self.p2_max)

    def run(self):

        print(f"---- running grid search for {self.exp_name} ----")
        if self.freeze_p2:
            while self.gamma1 > self.gamma_1_stop:
                self.update_scores()
                self.update_grid()
            print(" Done ")
        else:
            while (self.gamma1 > self.gamma_1_stop) or (self.gamma2 > self.gamma_2_stop):
                self.update_scores()
                self.update_grid()
            print(" Done ")


def get_time_str():
    now = datetime.now()
    return(now.strftime("%Y/%m/%d, %H:%M:%S"))