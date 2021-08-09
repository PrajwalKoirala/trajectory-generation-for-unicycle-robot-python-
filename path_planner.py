from sys import path
path.append(r"/home/prajwal/Libraries/Casadi_Py_Lib/casadi_v3.5.5")
from casadi import *
import matplotlib.pyplot as plt;
import numpy as np;
import time;

class Traj_Planner(object):
    def __init__(self, opti, prbls):
        self.opti = opti;
        self.N = prbls['N']; self.dt = prbls['dt']; self.No_obs = prbls['No_obs'];
        N = self.N;
        x = self.opti.variable(1, N+1); y = self.opti.variable(1, N+1); theta = self.opti.variable(1, N+1); 
        self.X = vertcat(x, y, theta);
        v = self.opti.variable(1, N); w = self.opti.variable(1, N);
        self.U = vertcat(v,w);
        self.X_0 = self.opti.parameter(3,1);
        self.X_f = self.opti.parameter(3,1);
        self.X_dot = self.f_x_u();
        self.subject_to_dynamic_constraints();
        self.subject_to_static_constraints();   
        self.Obs_pos = self.opti.parameter(2,self.No_obs); self.Obs_rad = self.opti.parameter(1,self.No_obs);
        cost_obstacles = self.Obstacle();
        self.opti.minimize( 0.3*sum2((x-self.X_f[0])**2) + 0.3*sum2((y-self.X_f[1])**2) +  0.2*sum2((theta-self.X_f[2])**2) + 0.2*sum1(sum2((self.U)**2)) +  cost_obstacles);
        p_opts = {"expand":True};
        s_opts = {"max_iter": 1000};
        opti.solver("ipopt", p_opts, s_opts);

    def return_solution(self, X_now, X_ref, Obstacles,  Initialization=None):
        if not(Initialization is None):
            self.opti.set_initial(self.U,Initialization['U']); self.opti.set_initial(self.X, Initialization['X']);
        if self.No_obs>0:
            self.opti.set_value(self.Obs_pos, Obstacles[0:2,0:self.No_obs]); 
            self.opti.set_value(self.Obs_rad, Obstacles[2,0:self.No_obs]);
        self.opti.set_value(self.X_0, X_now); self.opti.set_value(self.X_f, X_ref);
        sol = self.opti.solve();
        return sol;


    def f_x_u(self):
        x = MX.sym('x',3);
        u = MX.sym('u',2);
        f_x_u = vertcat(u[0]*cos(x[2]), u[0]*sin(x[2]), u[1]);
        X_dot_act = Function('X_dot',[x, u],[f_x_u]); 
        return X_dot_act;

    def subject_to_static_constraints(self):
        self.opti.subject_to(self.X[:,0]==self.X_0);

    def subject_to_dynamic_constraints(self):
        for k in range(self.N):
            self.opti.subject_to( self.X[:,k+1]== self.X[:,k] + self.dt*self.X_dot(self.X[:,k],self.U[:,k]) );


    def Obstacle(self):
        obst_cost = 0;
        if self.No_obs>0:
            No_obs = self.No_obs; 
            Obs_pos = self.Obs_pos; Obs_rad = self.Obs_rad;        
            for i in range(No_obs):
                h =  log( ((self.X[0,:]-Obs_pos[0,i])/Obs_rad[i])**2 + ((self.X[1,:]-Obs_pos[1,i])/Obs_rad[i])**2  );
                self.opti.subject_to(h>0);  
                obst_cost = obst_cost + sum2(exp(6*exp(-h)));
        return obst_cost;



def main():
    opti=casadi.Opti();  
    dt = 0.05;
    prbls = {'N':20, 'dt':dt, 'No_obs':3};
    simtime = 6;
    traj = Traj_Planner(opti, prbls);
    print(traj.opti);
    X_now = [0,0,0];
    X_ref = [10,10,pi];
    Obstacles = horzcat([4,4,1],[8,4,1], [8,8,1]);
    #MPC loop:
    iter = int(simtime/prbls['dt']);
    X_es = horzcat(X_now, DM(3, iter));
    U_s = DM(2,iter);
    sol = traj.return_solution(X_now, X_ref, Obstacles);
    Initialization = {'X':sol.value(traj.X), 'U':sol.value(traj.U)};
    tic = time.time();
    for i in range(iter):
        sol = traj.return_solution(X_now, X_ref, Obstacles, Initialization);
        u_set = sol.value(traj.U); x_set = sol.value(traj.X);
        Initialization = {'X':x_set, 'U':u_set};
        # X_now = DM(X_now) + dt*traj.X_dot(X_now,u_set[:,0]).full();
        X_now = x_set[:,1];
        X_es[:,i+1] = X_now; U_s[:,i] = u_set[:,0];
    toc = time.time(); print(toc-tic,'seconds elapsed')
    x = X_es[0,:];
    y = X_es[1,:];
    theta = X_es[2,:];
    t = [i*dt for i in range(iter+1)];
    plt.plot(x.T,y.T); plt.axis('square');
    for i in range(prbls['No_obs']):
        theta = [k/100*2*pi for k in range(101)];plt.plot(Obstacles[0,i] + Obstacles[2,i]*cos(theta), Obstacles[1,i] + Obstacles[2,i]*sin(theta));
    plt.title('x-y trajectory'); plt.show();
    plt.plot(t, x.T); plt.plot(t, y.T); plt.show();



if __name__ == '__main__':
    main();
