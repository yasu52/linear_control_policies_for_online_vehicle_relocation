import os
import pickle
import time
import math

import pulp 
import pandas as pd
from scipy import stats

from tqdm import tqdm
import settings as st



CURRENT_PATH = os.getcwd()
INPUT_TRIP_PATH = None
INTERMEDIATE_TRAIN_PATH = None
INTERMEDIATE_TEST_PATH = None


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)


def set_dir_path():

  global CURRENT_PATH
  global INPUT_TRIP_PATH
  global INPUT_STATION_PATH
  global INTERMEDIATE_TRAIN_PATH
  global INTERMEDIATE_TEST_PATH
  global OUTPUT_SUMMARY_PATH

  INPUT_TRIP_PATH = os.path.join('../../', st.input_trip_path)
  INPUT_STATION_PATH = os.path.join('../../', st.input_station_path)
  INTERMEDIATE_TRAIN_PATH = os.path.join('../', st.intermediate_train_path)
  INTERMEDIATE_TEST_PATH = os.path.join('../', st.intermediate_test_path)
  OUTPUT_SUMMARY_PATH = os.path.join('../../', st.output_summary_path)

  print('CURRENT PATH:{}'.format(CURRENT_PATH),flush=True)
  print('INPUT TRIP PATH:{}'.format(INPUT_TRIP_PATH),flush=True)
  print('INPUT STATION PATH:{}'.format(INPUT_STATION_PATH),flush=True)
  print('INTERMEDIATE TRAIN PATH:{}'.format(INTERMEDIATE_TRAIN_PATH),flush=True)
  print('INTERMEDIATE TEST PATH:{}'.format(INTERMEDIATE_TEST_PATH),flush=True)
  print('OUTPUT SUMMARY PATH:{}'.format(INTERMEDIATE_TEST_PATH),flush=True)

  os.makedirs(INTERMEDIATE_TRAIN_PATH, exist_ok=True)
  os.makedirs(INTERMEDIATE_TEST_PATH, exist_ok=True)
  os.makedirs(OUTPUT_SUMMARY_PATH, exist_ok=True)

  return


def read_trip_data(s_trip_file, e_trip_file):

  s_trip = pickle_load(os.path.join(INPUT_TRIP_PATH, s_trip_file))
  e_trip = pickle_load(os.path.join(INPUT_TRIP_PATH, e_trip_file))

  return s_trip, e_trip



class MobilitySolver_MPC():

      def __init__(self):

        self.station_num = st.station_num
        self.time_num = st.time_num

        return


      def read_station_data(self):

          self.cluster_dfm = pd.read_csv(os.path.join(INPUT_STATION_PATH, 'cluster.csv'))
          self.duration_dfm = pd.read_csv(os.path.join(INPUT_STATION_PATH, 'trip_duration.csv'))

          return


      def read_train_data(self):

        self.s_trip, self.e_trip = read_trip_data(st.s_trip_train_file, st.e_trip_train_file)

        return


      def read_test_data(self):

        self.s_trip, self.e_trip = read_trip_data(st.s_trip_test_file, st.e_trip_test_file)

        return


      def set_data(self):

        self.reb_span = st.reb_span
        self.reb_start = st.reb_start
        self.reb_end = st.reb_end
        self.scenario_num = int(len(self.s_trip)/self.time_num/self.station_num/self.station_num)
        self.control_name = st.control_name

        #Sets
        self.D = list(range(1, self.station_num+1))
        self.S = list(range(1, self.scenario_num+1))
        self.T = list(range(1, self.time_num+1))
        self.T_ = list(range(1, self.time_num+2))
        self.T2reb = list((self.reb_start*12)+time*self.reb_span*12+1 for time in range(round((self.reb_end-self.reb_start)/self.reb_span)+1))

        self.STJ = [(s,t,j) for j in self.D for t in self.T_ for s in self.S]
        self.STIJ = [(s,t,i,j) for j in self.D for i in self.D for t in self.T for s in self.S]
        self.TIJ = [(t,i,j) for j in self.D for i in self.D for t in self.T]
        self.SJ = [(s,j) for j in self.D for s in self.S]
        self.IJ = [(i,j) for j in self.D for i in self.D]
        self.ST2rebIJ = [(s,t,i,j) for j in self.D for i in self.D for t in self.T2reb for s in self.S]
        self.T2rebIJ = [(t,i,j) for j in self.D for i in self.D for t in self.T2reb]
        self.ST2rebJ = [(s,t,j) for j in self.D for t in self.T2reb for s in self.S]

        #Parameters
        self.C2Max = {row.cluster_id:row.cluster_max for row in self.cluster_dfm.itertuples()}
        self.C2Min = {row.cluster_id:row.cluster_min for row in self.cluster_dfm.itertuples()}
        self.Z2I = {row.cluster_id:row.cluster_ini for row in self.cluster_dfm.itertuples()}

        self.R2Max = st.R
        self.Z2Max = sum(self.Z2I.values())

        self.alpha = st.alpha
        self.beta = st.beta
        self.gamma = st.gamma
        self.mu = 0.50

        self.Tau = {}
        for i in self.D:
          for j in self.D:
            if i == j:
              self.Tau[i,j] = 0
            else:
              self.Tau[i,j] = self.duration_dfm.iat[i-1,j-1]
        self.Tau_pol = st.Tau_pol

        return


      def create_params(self):

        delta = 5

        self.q = {}
        for i in self.D:
          for j in self.D:
            if i != j:
              self.q[i,j] = 1-1/math.e**(delta/(self.Tau[i,j]*5))
            else:
              self.q[i,j] = 1

        self.p = {}
        for t in range(0,24):
          for i in self.D:
            trip_from_i_sum = 0
            for s in self.S:
              for n in range(12*t+1,12*t+13):
                for h in self.D:
                  trip_from_i_sum += self.s_trip[s,n,i,h]
            for j in self.D:
              trip_i_to_j = 0
              for s in self.S:
                for n in range(12*t+1,12*t+13):
                  trip_i_to_j += self.s_trip[s,n,i,j]
              if trip_from_i_sum != 0:
                self.p[t,i,j] = trip_i_to_j/trip_from_i_sum
              else:
                self.p[t,i,j] = 0

        self.l = {}
        for t in range(0,24):
          for i in self.D:
            trip_i_sum = 0
            for s in self.S:
              for n in range(12*t+1,12*t+13):
                for j in self.D:
                  trip_i_sum += self.s_trip[s,n,i,j]
            self.l[t,i] = trip_i_sum/self.scenario_num/12

        return


      def const_allocation(self, R_s, s, t, R2Max):
          sum_R_t = sum(R_s[s,t,i,j] for i in self.D for j in self.D)
          if sum_R_t <= R2Max:return R_s
          X = sum_R_t/R2Max
          L = []
          sum_R_int = 0
          for i in self.D:
              for j in self.D:
                  if R_s[s,t,i,j] < 1:continue
                  R_stij_int = int(R_s[s,t,i,j]/X)
                  sum_R_int += R_stij_int
                  L.append([((R_s[s,t,i,j]/X)-R_stij_int),i,j])
                  R_s[s,t,i,j] = R_stij_int
          L_sort = sorted(L, key=lambda x: -x[0])
          for l in L_sort:
              i = l[1]
              j = l[2]
              R_s[s,t,i,j] += 1
              sum_R_int += 1
              if sum_R_int == R2Max:
                  break
          return R_s
      

      def solve_expect(self):

          time_start_all = time.time()

          #Conditions
          Z = {}
          A = {}
          R = {}
          V = {}
          for stj in self.STJ:
            Z[stj] = 0 #t <= 289
            A[stj] = 0 #t <= 289
          for stij in self.STIJ:
            V[stij] = 0 #t <= 288
          for stij in self.ST2rebIJ:
            R[stij] = 0 #97 <= t <= 241

          #The initial number of cehicles
          for s in self.S:
            for j in self.D:
              Z[s,1,j] = self.Z2I[j]

          def func_r_s_T2hour(s,t,i,j):
            if t - self.Tau[i,j] in self.T2reb:
              t_past = t - self.Tau[i,j]
              return R[s,t_past,i,j]
            else:
              return 0

          for s in tqdm(self.S):
            time_start_s = time.time()

            for t in self.T:

              if t not in self.T2reb:

                for j in self.D:
                  Z[s,t+1,j] = Z[s,t,j] - sum(self.s_trip[s,t,j,i] for i in self.D) + sum(self.e_trip[s,t,i,j] for i in self.D) + sum(func_r_s_T2hour(s,t,i,j) for i in self.D)

                for i in self.D:
                  for j in self.D:
                    for k in range(1,t+1):
                      V[s,t,i,j] += (self.s_trip[s,k,i,j]-self.e_trip[s,k,i,j])

                for j in self.D:
                  if Z[s,t,j] > self.C2Max[j]:
                    A[s,t,j] = Z[s,t,j] - self.C2Max[j]
                  elif Z[s,t,j] < self.C2Min[j]:
                    A[s,t,j] = self.C2Min[j] - Z[s,t,j]

              if t == self.time_num:
                for j in self.D:
                  if Z[s,t+1,j] > self.C2Max[j]:
                    A[s,t+1,j] = Z[s,t+1,j] - self.C2Max[j]
                  elif Z[s,t,j] < self.C2Min[j]:
                    A[s,t+1,j] = self.C2Min[j] - Z[s,t+1,j]

              ############################### Optimization Module ###############################
              elif t in self.T2reb:

                KT = [k for k in range(t,self.time_num+1)]
                KT_ = [k for k in range(t,self.time_num+2)]
                KT2reb= [(k*12)+1 for k in range(int((t-1)/12),21)]

                J = [j for j in self.D]
                KTJ = [(t,j) for j in self.D for t in KT_]
                KTIJ = [(t,i,j) for j in self.D for i in self.D for t in KT]
                KT2rebIJ = [(t,i,j) for j in self.D for i in self.D for t in KT2reb]

                for i in self.D:
                  for j in self.D:
                    for k in range(1,t+1):
                      V[s,t,i,j] += (self.s_trip[s,k,i,j]-self.e_trip[s,k,i,j])

                problem = pulp.LpProblem('BikeShare',pulp.LpMinimize)

                #Decision Variables
                z = pulp.LpVariable.dicts('z', KTJ, cat='Continuous')
                a = pulp.LpVariable.dicts('a', KTJ,lowBound=0, cat='Continuous')
                v = pulp.LpVariable.dicts('v', KTIJ,lowBound=0, cat='Continuous')
                r_s = pulp.LpVariable.dicts('r_s', KT2rebIJ,lowBound=0, cat='Continuous')

                x = pulp.LpVariable.dicts('x', J,lowBound=0, cat='Continuous')
                y = pulp.LpVariable.dicts('y', J,lowBound=0, cat='Continuous')


                #Objective Function
                problem += self.alpha*pulp.lpSum(r_s[k,i,j] for j in self.D for i in self.D for k in KT2reb) + self.beta*pulp.lpSum(a[k,j] for j in self.D for k in KT_) + self.gamma*pulp.lpSum(x[j]+y[j] for j in self.D)

                #Absolute Value
                for j in self.D:
                  problem += z[self.time_num+1,j]-Z[s,1,j] == x[j]-y[j]

                #Initialize Station
                for i in self.D:
                  for j in self.D:
                    problem += v[t,i,j] == V[s,t,i,j]
                    problem += z[t,j] == Z[s,t,j]

                #Trip
                lambda_start = {}
                for k in KT:
                  for i in self.D:
                    lambda_start[k,i] = self.l[(k-1)//12,i]

                #flow conservation
                for k in KT:
                  for i in self.D:
                    for j in self.D:
                      if k in self.T2reb:
                        problem += v[k+1,i,j] == (1-self.q[i,j])*v[k,i,j]+self.p[(k-1)//12,i,j]*lambda_start[k,i]+r_s[k,i,j]
                      else:
                        if k != self.time_num:
                        #if k != t+23:
                          problem += v[k+1,i,j] == (1-self.q[i,j])*v[k,i,j]+self.p[(k-1)//12,i,j]*lambda_start[k,i]

                #station conservation
                for k in KT_:
                  for j in self.D:
                    if k in self.T2reb:
                    #if k in T2hour and k != t+24:
                      problem += z[k+1,j] == z[k,j]+sum(self.q[i,j]*v[k,i,j] for i in self.D)-lambda_start[k,j]-sum(r_s[k,j,h] for h in self.D)
                    else:
                      if k != self.time_num+1:
                      #if k != t+24:
                        problem += z[k+1,j] == z[k,j]+sum(self.q[i,j]*v[k,i,j] for i in self.D)-lambda_start[k,j]

                #violations of station capacity
                for k in KT_:
                    for j in self.D:
                        problem += z[k,j] <= self.C2Max[j] + a[k,j]
                        problem += z[k,j] >= self.C2Min[j] - a[k,j]
                        
                #relocation limit in each time period
                for k in KT:
                  if k in self.T2reb:
                    problem += pulp.lpSum(r_s[k,i,j] for i in self.D for j in self.D) <= self.R2Max

                time_start_t = time.time()
                status = problem.solve()
                time_stop_t = time.time()
                print("===",time_stop_t-time_start_t,"===",s,"===",t)
                print(pulp.LpStatus[status])
                print(pulp.value(problem.objective))

                for i in self.D:
                  for j in self.D:
                    R[s,t,i,j] = round(r_s[t,i,j].value())
                for j in self.D:
                  Z[s,t+1,j] = Z[s,t,j] - sum(self.s_trip[s,t,j,i] for i in self.D) - sum(R[s,t,j,h] for h in self.D) + sum(self.e_trip[s,t,i,j] for i in self.D)
                for j in self.D:
                  A[s,t,j] = a[t,j].value()

                R = self.const_allocation(R, s, t, self.R2Max)

            time_stop_s = time.time()
            print("===",time_stop_s-time_start_s,"===",s)

          time_stop_all = time.time()

          sum_A = 0
          for s in self.S:
            for j in self.D:
              for t in self.T_:
                sum_A += A[s,t,j]

          sum_R = 0
          for s in self.S:
            for i in self.D:
              for j in self.D:
                for t in self.T2reb:
                  sum_R += R[s,t,i,j]

          sum_XY = 0
          for s in self.S:
              for j in self.D:
                  sum_XY += abs(Z[s,self.time_num+1,j]-Z[s,1,j])

          self.R_s_sce = sum_R/len(self.S)
          self.A_sce = sum_A/len(self.S)
          self.XY_sce = sum_XY/self.scenario_num
          self.time = time_stop_all-time_start_all

          return R,A,Z


      def solve_robust(self):

          time_start_all = time.time()

          Z = {}
          A = {}
          R = {}
          V = {}
          for stj in self.STJ:
            Z[stj] = 0 
            A[stj] = 0 
          for stij in self.STIJ:
            V[stij] = 0 
          for stij in self.ST2rebIJ:
            R[stij] = 0 

          for s in self.S:
            for j in self.D:
              Z[s,1,j] = self.Z2I[j]

          def func_r_s_T2hour(s,t,i,j):
            if t - self.Tau[i,j] in self.T2reb:
              t_past = t - self.Tau[i,j]
              return R[s,t_past,i,j]
            else:
              return 0

          for s in tqdm(self.S):
            time_start_s = time.time()

            for t in self.T:

              if t not in self.T2reb:

                for j in self.D:
                  Z[s,t+1,j] = Z[s,t,j] - sum(self.s_trip[s,t,j,i] for i in self.D) + sum(self.e_trip[s,t,i,j] for i in self.D) + sum(func_r_s_T2hour(s,t,i,j) for i in self.D)

                for i in self.D:
                  for j in self.D:
                    for k in range(1,t+1):
                      V[s,t,i,j] += (self.s_trip[s,k,i,j]-self.e_trip[s,k,i,j])

                for j in self.D:
                  if Z[s,t,j] > self.C2Max[j]:
                    A[s,t,j] = Z[s,t,j] - self.C2Max[j]
                  elif Z[s,t,j] < self.C2Min[j]:
                    A[s,t,j] = self.C2Min[j] - Z[s,t,j]

              if t == self.time_num:
                for j in self.D:
                  if Z[s,t+1,j] > self.C2Max[j]:
                    A[s,t+1,j] = Z[s,t+1,j] - self.C2Max[j]
                  elif Z[s,t,j] < self.C2Min[j]:
                    A[s,t+1,j] = self.C2Min[j] - Z[s,t+1,j]

              elif t in self.T2reb:

                KT = [k for k in range(t,self.time_num+1)]
                KT_ = [k for k in range(t,self.time_num+2)]
                KT2reb = [(k*12)+1 for k in range(int((t-1)/12),21)]

                J = [j for j in self.D]
                KTJ = [(t,j) for j in self.D for t in KT_]
                KTIJ = [(t,i,j) for j in self.D for i in self.D for t in KT]
                KT2rebIJ = [(t,i,j) for j in self.D for i in self.D for t in KT2reb]

                for i in self.D:
                  for j in self.D:
                    for k in range(1,t+1):
                      V[s,t,i,j] += (self.s_trip[s,k,i,j]-self.e_trip[s,k,i,j])

                problem = pulp.LpProblem('BikeShare',pulp.LpMinimize)

                z_min = pulp.LpVariable.dicts('z_min', KTJ, cat='Continuous')
                z_max = pulp.LpVariable.dicts('z_max', KTJ, cat='Continuous')
                a_min = pulp.LpVariable.dicts('a_min', KTJ,lowBound=0, cat='Continuous')
                a_max = pulp.LpVariable.dicts('a_max', KTJ,lowBound=0, cat='Continuous')
                v_min = pulp.LpVariable.dicts('v_min', KTIJ,lowBound=0, cat='Continuous')
                v_max = pulp.LpVariable.dicts('v_max', KTIJ,lowBound=0, cat='Continuous')
                r_s = pulp.LpVariable.dicts('r_s', KT2rebIJ,lowBound=0, cat='Continuous')

                x_min = pulp.LpVariable.dicts('x_min', J,lowBound=0, cat='Continuous')
                x_max = pulp.LpVariable.dicts('x_max', J,lowBound=0, cat='Continuous')
                y_min = pulp.LpVariable.dicts('y_min', J,lowBound=0, cat='Continuous')
                y_max = pulp.LpVariable.dicts('y_max', J,lowBound=0, cat='Continuous')

                problem += self.alpha*pulp.lpSum(r_s[k,i,j] for j in self.D for i in self.D for k in self.KT2reb) + self.beta*pulp.lpSum((a_min[k,j]+a_max[k,j]) for j in self.D for k in KT_) + self.gamma*(pulp.lpSum(x_min[j]+y_min[j] for j in self.D)+pulp.lpSum(x_max[j]+y_max[j] for j in self.D)) 

                for j in self.D:
                  problem += z_min[self.time_num+1,j]-Z[s,1,j] == x_min[j]-y_min[j]                
                  problem += z_max[self.time_num+1,j]-Z[s,1,j] == x_max[j]-y_max[j]   

                for i in self.D:
                  for j in self.D:
                    problem += v_min[t,i,j] == V[s,t,i,j]
                    problem += v_max[t,i,j] == V[s,t,i,j]
                    problem += z_min[t,j] == Z[s,t,j]
                    problem += z_max[t,j] == Z[s,t,j]

                lambda_start_min = {}
                lambda_start_max = {}
                for k in KT:
                  for i in self.D:
                    lower, upper = stats.poisson.interval(alpha=self.mu, mu=self.l[(k-1)//12,i]) # 信頼係数=0.50
                    lambda_start_min[k,i] = lower
                    lambda_start_max[k,i] = upper

                for k in KT:
                  for i in self.D:
                    for j in self.D:
                      if k in self.T2reb:
                        problem += v_min[k+1,i,j] == (1-self.q[i,j])*v_min[k,i,j]+self.p[(k-1)//12,i,j]*lambda_start_min[k,i]+r_s[k,i,j]
                        problem += v_max[k+1,i,j] == (1-self.q[i,j])*v_max[k,i,j]+self.p[(k-1)//12,i,j]*lambda_start_max[k,i]+r_s[k,i,j]
                      else:
                        if k != self.time_num:
                        #if k != t+23:
                          problem += v_min[k+1,i,j] == (1-q[i,j])*v_min[k,i,j]+p[(k-1)//12,i,j]*lambda_start_min[k,i]
                          problem += v_max[k+1,i,j] == (1-q[i,j])*v_max[k,i,j]+p[(k-1)//12,i,j]*lambda_start_max[k,i]

                for k in KT_:
                  for j in self.D:
                    if k in self.T2reb:
                    #if k in T2hour and k != t+24:
                      problem += z_min[k+1,j] == z_min[k,j]+sum(self.q[i,j]*v_min[k,i,j] for i in self.D)-lambda_start_min[k,j]-sum(r_s[k,j,h] for h in self.D)
                      problem += z_max[k+1,j] == z_max[k,j]+sum(self.q[i,j]*v_max[k,i,j] for i in self.D)-lambda_start_max[k,j]-sum(r_s[k,j,h] for h in self.D)
                    else:
                      if k != self.time_num+1:
                      #if k != t+24:
                        problem += z_min[k+1,j] == z_min[k,j]+sum(self.q[i,j]*v_min[k,i,j] for i in self.D)-lambda_start_min[k,j]
                        problem += z_max[k+1,j] == z_max[k,j]+sum(self.q[i,j]*v_max[k,i,j] for i in self.D)-lambda_start_max[k,j]

                for k in KT_:
                    for j in self.D:
                        problem += z_min[k,j] <= self.C2Max[j] + a_min[k,j]
                        problem += z_max[k,j] <= self.C2Max[j] + a_max[k,j]
                        problem += z_min[k,j] >= self.C2Min[j] - a_min[k,j]
                        problem += z_max[k,j] >= self.C2Min[j] - a_max[k,j]

                for k in KT:
                  if k in self.T2reb:
                    problem += pulp.lpSum(r_s[k,i,j] for i in self.D for j in self.D) <= self.R2Max

                time_start_t = time.time()
                status = problem.solve()
                time_stop_t = time.time()
                print("===",time_stop_t-time_start_t,"===",s,"===",t)
                print(pulp.LpStatus[status])
                print(pulp.value(problem.objective))

                for i in self.D:
                  for j in self.D:
                    R[s,t,i,j] = round(r_s[t,i,j].value())

                for j in self.D:
                  Z[s,t+1,j] = Z[s,t,j] - sum(self.s_trip[s,t,j,i] for i in self.D) - sum(R[s,t,j,h] for h in self.D) + sum(self.e_trip[s,t,i,j] for i in self.D)

                for j in self.D:
                  if Z[s,t,j] > self.C2Max[j]:
                    A[s,t,j] = Z[s,t,j] - self.C2Max[j]
                  elif Z[s,t,j] < self.C2Min[j]:
                    A[s,t,j] = self.C2Min[j] - Z[s,t,j]

                R = self.const_allocation(R, s, t, self.R2Max)

            time_stop_s = time.time()
            print("===",time_stop_s-time_start_s,"===",s)

          time_stop_all = time.time()

          sum_A = 0
          for s in self.S:
            for j in self.D:
              for t in self.T_:
                sum_A += A[s,t,j]

          sum_R = 0
          for s in self.S:
            for i in self.D:
              for j in self.D:
                for t in self.T2reb:
                  sum_R += R[s,t,i,j]

          sum_XY = 0
          for s in self.S:
              for j in self.D:
                  sum_XY += abs(Z[s,self.time_num+1,j]-Z[s,1,j])

          self.R_s_sce = sum_R/len(self.S)
          self.A_sce = sum_A/len(self.S)
          self.XY_sce = sum_XY/self.scenario_num
          self.time = time_stop_all-time_start_all

          return R,A,Z


      def write_test_result(self, R, A, Z):

        pickle_dump(R, os.path.join(INTERMEDIATE_TEST_PATH, f'r_s.pickle'))
        pickle_dump(A, os.path.join(INTERMEDIATE_TEST_PATH, f'a.pickle'))
        pickle_dump(Z, os.path.join(INTERMEDIATE_TEST_PATH, f'z.pickle'))

        return



def write_summary(ms_mpc):

  with open(os.path.join(OUTPUT_SUMMARY_PATH,'summary.txt'), 'w') as f:

    f.write('MPC RelVeh    :%f\n' % ms_mpc.R_s_sce)
    f.write('MPC StaCap    :%f\n' % ms_mpc.A_sce)
    f.write('MPC IniCon    :%f\n' % ms_mpc.XY_sce)
    f.write('MPC Obj       :{}\n\n'.format(ms_mpc.R_s_sce+ms_mpc.A_sce+ms_mpc.XY_sce))

    f.write('MPC Elapsed Time  :{}\n'.format(ms_mpc.time))


  with open(os.path.join(OUTPUT_SUMMARY_PATH,'settings.txt'), 'w') as f:

    f.write('input trip path   :{}\n'.format(st.input_trip_path))
    f.write('input station path:{}\n'.format(st.input_station_path))
    f.write('start trip train  :{}\n'.format(st.s_trip_train_file))
    f.write('end   trip train  :{}\n'.format(st.e_trip_train_file))
    f.write('start trip test   :{}\n'.format(st.s_trip_test_file))
    f.write('end   trip test   :{}\n\n'.format(st.e_trip_test_file))

    f.write('intermediate train path   :{}\n'.format(st.intermediate_train_path))   
    f.write('intermediate test  path   :{}\n\n'.format(st.intermediate_test_path))

    f.write('output summary path   :{}\n\n'.format(st.output_summary_path))

    f.write('station num     :{}\n'.format(st.station_num))
    f.write('time num        :{}\n'.format(st.time_num))
    f.write('rebalance span  :{}\n'.format(st.reb_span))
    f.write('rebalance start :{}\n'.format(st.reb_start))
    f.write('rebalance end   :{}\n'.format(st.reb_end))
    f.write('control name    :{}\n'.format(st.control_name))

  return



def main():

  set_dir_path()

  ms_mpc = MobilitySolver_MPC()
  ms_mpc.read_station_data()

  ms_mpc.read_train_data()
  ms_mpc.set_data()
  ms_mpc.create_params()

  ms_mpc.read_test_data()
  ms_mpc.set_data()

  if ms_mpc.control_name == 'mpc_expect':
    print('===== Expect =====')
    R, A, Z = ms_mpc.solve_expect()
  elif ms_mpc.control_name == 'mpc_robust':
    print('===== Robust =====')
    R, A, Z = ms_mpc.solve_robust()

  ms_mpc.write_test_result(R, A, Z)

  write_summary(ms_mpc)

  return


if __name__ == '__main__':

  main()
