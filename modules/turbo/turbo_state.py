import torch
import math

class TurboState():

    def __init__(self,
                 dim,
                 batch_size,
                 tr_init_x_norm,
                 tr_init_y,
                 tr_init_df,
                 tr_id=None,
                 length=0.8,
                 length_min=0.5**7,
                 length_max=1.6,
                 failure_counter=0,
                 success_counter=0,
                 success_tolerance=3,
                 ):
        self.dim=dim
        self.batch_size=batch_size
        self.length=length
        self.length_min=length_min
        self.length_max=length_max
        self.failure_counter=failure_counter
        self.success_counter=success_counter
        self.success_tolerance=success_tolerance
        self.tr_x_norm=tr_init_x_norm
        self.tr_y=tr_init_y
        self.tr_df=tr_init_df
        self.tr_id=tr_id

        #self.failure_tolerance = math.ceil(
                    #max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
                #)
        self.failure_tolerance=7

        self.best_value=float("-inf")
        self.tr_x_best_norm=None
        self.tr_x_best=None

        self.restart_triggered=False

        self.tr_ub=None
        self.tr_lb=None

        self.tr_gp_model=None


    def update_best_value(self,best_value):
        self.best_value=best_value


    def update_state(self,Y_next):
        if max(Y_next) > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length = max(self.length/2.0,self.length_min)
            self.failure_counter = 0

        self.best_value = max(self.best_value, max(Y_next).item())
        if self.length < self.length_min:
            self.restart_triggered = True
        

