import numpy as np
import math
import pandas as pd

class ValidateCoarseToFine():
    def __init__(self, score, dir_name, exp_name, p1_init=1, p2_init=10, gamma1=4, gamma2=4, decay_factor=0.5, grid_size=3, freeze_p2=False, verbose=True, gamma_stop=1.01, p1_max=float('inf'), p2_max=float('inf')):

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
        self.exp_name = f"ValidationScores_{exp_name}"

        # upper-bound, if any
        self.p1_max = p1_max
        self.p2_max = p2_max

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
        cols = ["exp", "p1", "p2", "psnr", "ssim", "niter"]
        dtype = ["string", "float", "float", "float", "float","float"]
        self.scores= pd.concat([pd.Series(name=col, dtype=dt) for col, dt in zip(cols, dtype)], axis=1)


        # verbose
        self.verbose = verbose

        # only optimize over p1
        self.freeze_p2 = freeze_p2

        self.gamma_stop = gamma_stop


    # update scores on the local grid
    def update_scores(self):
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
                
                # check if psnr already computed for (almost) the same parameters
                df_t = self.scores[((1 - self.scores["p1"]/p1).abs()<(1e-4))]
                df_t = df_t[(df_t["p2"]/p2 - 1).abs()<(1e-4)]
        
                if df_t.shape[0] > 0:
                    psnr_t = df_t.iloc[0, 3]
                    ssim_t = df_t.iloc[0, 4]
                    niter_t = df_t.iloc[0, 5]
                    
                    self.psnr_grid[k, j] = psnr_t
                    self.ssim_grid[k, j] = ssim_t
                    print(f"** p1={p1}, p2={p2} **")
                    print(" skipping the computation ")
                    print(f" (found values for p1={df_t.iloc[0, 1]}, p2={df_t.iloc[0, 2]} **")
                elif (p1 > self.p1_max):
                    print(f" skipping value since p1 greater than {self.p1_max}")
                elif (p2 > self.p2_max):
                    print(f" skipping value since p2 greater than {self.p2_max}")
                else:
                    psnr_t, ssim_t,  niter_t = self.score(p1, p2)
                    
                    self.psnr_grid[k, j] = psnr_t
                    self.ssim_grid[k, j] = ssim_t

                    df2 = pd.DataFrame([[self.exp_name, p1, p2, psnr_t, ssim_t, niter_t]], columns=self.scores.columns)
                    self.scores = pd.concat((self.scores, df2))
                    self.save_scores()
                    
        
                if self.verbose:
                    print(f"PSNR {psnr_t:.3f} __ running max {self.scores['psnr'].max():.3f}")
                    print(f"SSIM {ssim_t:.3f} __ running max {self.scores['ssim'].max():.3f}")
    
    def save_scores(self):
        self.scores.to_csv(f"{self.dir_name}/{self.exp_name}.csv")

    # update grid properties based on the scores
    def update_grid(self):
        ind = np.unravel_index(np.argmax(self.psnr_grid, axis=None), self.psnr_grid.shape)

        if ind[0] == 0:
            self.p1 = self.p1 * (self.gamma1)**(-(self.grid_size//2))
            print("! lower border hit for p1 !")
        elif ind[0] == self.grid_size - 1:
            
            self.p1 = self.p1 * (self.gamma1)**(self.grid_size//2)
            print("! upper border hit for p1 !")
        else:
            # find new center
            p1_new_center = self.p1*(self.gamma1)**ind[0]
            # reduce scale
            self.gamma1 = (self.gamma1**self.decay_factor)
            # update left point
            self.p1 = p1_new_center*self.gamma1**(-(self.grid_size//2))
            print(f"new p1 {self.p1}")
            print(f"new gamma1 {self.gamma1}")


        if not self.freeze_p2:
            if ind[1] == 0:
                print("! lower border hit for p2 !")
                self.p2 = self.p2 * (self.gamma2)**(-(self.grid_size//2))
                
            elif ind[1] == self.grid_size - 1:
                print("! upper border hit for p2 !")
                self.p2 = self.p2 * (self.gamma2)**(self.grid_size//2)
                
            else:
                # find new center
                p2_new_center = self.p2*(self.gamma2)**ind[0]
                # reduce scale
                self.gamma2 = (self.gamma2**self.decay_factor)
                # update left point
                self.p2 = p2_new_center*self.gamma2**(-(self.grid_size//2))
                print(f"new p2 {self.p2}")
                print(f"new gamma2 {self.gamma2}")


    def run(self):
        if self.freeze_p2:
            while self.gamma1 > self.gamma_stop:
                self.update_scores()
                self.update_grid()
            print(" Done ")
        else:
            while (self.gamma1 > self.gamma_stop) and (self.gamma2 > self.gamma_stop):
                self.update_scores()
                self.update_grid()
            print(" Done ")

