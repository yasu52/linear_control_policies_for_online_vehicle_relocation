import os
import pickle
import time

import pandas as pd
import pulp 

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



class MobilitySolver:

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

    self.R = st.R
    self.Z2Max = sum(self.Z2I.values())

    self.alpha = st.alpha
    self.beta = st.beta
    self.gamma = st.gamma

    self.Tau = {}
    for i in self.D:
      for j in self.D:
        if i == j:
          self.Tau[i,j] = 0
        else:
          self.Tau[i,j] = self.duration_dfm.iat[i-1,j-1]
    self.Tau_pol = st.Tau_pol

    return


  def create_variables(self):
    self.a = pulp.LpVariable.dicts('a', self.STJ,lowBound=0, cat='Continuous')
    self.r_s = pulp.LpVariable.dicts('r_s', self.ST2rebIJ,lowBound=0, cat='Continuous')
    self.r = pulp.LpVariable.dicts('r', self.T2rebIJ, cat='Continuous')
    
    self.w = pulp.LpVariable.dicts('w', self.T2rebIJ,lowBound=0, cat='Continuous')

    self.x = pulp.LpVariable.dicts('x', self.SJ,lowBound=0, cat='Continuous')
    self.y = pulp.LpVariable.dicts('y', self.SJ,lowBound=0, cat='Continuous')

    self.z_i = pulp.LpVariable.dicts('z_i', self.D,lowBound=0, cat='Continuous')
    self.z = pulp.LpVariable.dicts('z', self.STJ, cat='Continuous')

    return


  def func_r_s_arr(self,s,t,i,j):

    if t - self.Tau[i,j] in self.T2reb:
      t_past = t - self.Tau[i,j]
      return self.r_s[s,t_past,i,j]
    else:
      return 0


  def create_constraints(self):

    for s in self.S:

      #The initial number of cehicles
      for j in self.D:
        self.problem += self.z_i[j] == self.Z2I[j]
        #self.problem += sum(self.z_i[j] for j in self.D) == self.Z2Max
        self.problem += self.z[s,1,j] == self.z_i[j]
      for j in self.D:
        self.problem += self.z[s,self.time_num+1,j] - self.z_i[j] == self.x[s,j] - self.y[s,j]

      #flow conservation constraint
      for t in self.T:
        for j in self.D:
          if t in self.T2reb:
            self.problem += self.z[s,t+1,j] == self.z[s,t,j] - sum(self.s_trip[s,t,j,i] for i in self.D) - sum(self.r_s[s,t,j,i] for i in self.D) + sum(self.e_trip[s,t,i,j] for i in self.D)
          else:
            self.problem += self.z[s,t+1,j] == self.z[s,t,j] - sum(self.s_trip[s,t,j,i] for i in self.D) + sum(self.e_trip[s,t,i,j] for i in self.D) + sum(self.func_r_s_arr(s,t,i,j) for i in self.D)
            
      #no relocation between same station
      for t in self.T2reb:
          for i in self.D:
              self.problem += self.r_s[s,t,i,i] == 0
              self.problem += self.w[t,i,i] == 0
              self.problem += self.r[t,i,i] == 0

      #relocation limit in each time period
      for t in self.T2reb:
        self.problem += pulp.lpSum(self.r_s[s,t,i,j] for i in self.D for j in self.D) <= self.R

      #violation of station capacity
      for t in self.T_:
          for j in self.D:
              self.problem += self.z[s,t,j] <= self.C2Max[j] + self.a[s,t,j]
              self.problem += self.z[s,t,j] >= self.C2Min[j] - self.a[s,t,j]

    return


  def culc_trip_dif(self):

    self.trip_dif = {}
    self.trip_d = {}
    for stj in self.ST2rebJ:
      self.trip_dif[stj] = 0
      self.trip_d[stj] = 0
    for s in self.S:
      for t in self.T2reb:
        for j in self.D:
          self.trip_dpt = 0
          self.trip_arr = 0
          for n in range(t-self.Tau_pol*12,t):
              for l in self.D:
                self.trip_dpt += int(self.s_trip[s,n,j,l])
                self.trip_arr += int(self.e_trip[s,n,l,j])
          self.trip_dif[s,t,j] = (self.trip_arr - self.trip_dpt)

    return 


  def no_reb(self):

    print('-- noreb --')

    for s in self.S:
      for t in self.T2reb:
        for i in self.D:
            for j in self.D:
              self.problem += self.r_s[s,t,i,j] == 0
              self.problem += self.r[t,i,j] == 0
              self.problem += self.w[t,i,j] == 0

    return


  def offline_reb(self):

    print('-- offline --')

    for s in self.S:
      for t in self.T2reb:
        for i in self.D:
            for j in self.D:
              self.problem += self.r_s[s,t,i,j] == self.r[t,i,j]

    return


  def lincon_reb(self):

    print('-- lincon --')

    self.culc_trip_dif()
    for s in self.S:
      for t in self.T2reb:
        for i in self.D:
          for j in self.D:
            self.problem += (self.r_s[s,t,i,j]-self.r_s[s,t,j,i]) == self.r[t,i,j] + self.w[t,i,j]*(self.trip_dif[s,t,i]/self.C2Max[i]-self.trip_dif[s,t,j]/self.C2Max[j])
    
    return


  def create_control_policy(self):

    if self.control_name == 'noreb':
      self.no_reb()
    elif self.control_name == 'offline':
      self.offline_reb()
    elif self.control_name == 'lincon':
      self.lincon_reb()
    else:
        pass

    return


  def create_objective(self):

    self.problem += self.alpha*pulp.lpSum(self.r_s[s,t,i,j] for j in self.D for i in self.D for t in self.T2reb for s in self.S)/self.scenario_num + self.beta*pulp.lpSum(self.a[s,t,j] for j in self.D for t in self.T_ for s in self.S)/self.scenario_num + self.gamma*pulp.lpSum(self.x[s,j]+self.y[s,j] for j in self.D for s in self.S)/self.scenario_num
    
    return


  def create_model(self):

    self.problem = pulp.LpProblem('BikeShare',pulp.LpMinimize)
    #Define Decision Variables
    self.create_variables()
    #Define Constraints
    self.create_constraints()
    #Define Control Policy
    self.create_control_policy()
    #Define Objective Function
    self.create_objective()

    return


  def preprocess_train(self):

    print('==== PREPROCESS TRAIN ====')

    print('--- Read Data ---')
    self.read_station_data()
    self.read_train_data()

    print('--- Set Data ---')
    self.set_data()

    print('--- Create Model ---')
    self.create_model()

    return


  def preprocess_test(self):

    print('==== PREPROCESS TEST ====')

    print('--- Read Data ---')
    self.read_station_data()
    self.read_test_data()

    print('--- Set Data ---')
    self.set_data()

    return


  def solve(self):

    print('==== SOLVE ====')

    self.start = time.time()
    self.status = self.problem.solve()
    self.end = time.time()

    self.status_name = pulp.LpStatus[self.status]
    self.elapsed_time = (self.end - self.start)
    self.objective = pulp.value(self.problem.objective)

    return


  def const_allocation(self):

    for s in self.S:
        for t in self.T2reb:
            sum_R_t = sum(self.R_s[s,t,i,j] for i in self.D for j in self.D)
            if sum_R_t <= self.R:continue
            X = sum_R_t/self.R
            L = []
            sum_R_int = 0
            for i in self.D:
                for j in self.D:
                    if self.R_s[s,t,i,j] < 1:continue
                    R_stij_int = int(self.R_s[s,t,i,j]/X)
                    sum_R_int += R_stij_int
                    L.append([((self.R_s[s,t,i,j]/X)-R_stij_int),i,j])
                    self.R_s[s,t,i,j] = R_stij_int
            L_sort = sorted(L, key=lambda x: -x[0])
            for l in L_sort:
                i = l[1]
                j = l[2]
                self.R_s[s,t,i,j] += 1
                sum_R_int += 1
                if sum_R_int == self.R:
                    break

    return


  def create_R_train(self):

    self.R_s = {}
    for s in self.S:
      for t in self.T2reb:
        for i in self.D:
          for j in self.D:
            self.R_s[s,t,i,j] = round(self.r_s[s,t,i,j].value())
    self.const_allocation()

    return


  def func_R_s_arr(self,s,t,i,j):

    if t - self.Tau[i,j] in self.T2reb:
      t_past = t - self.Tau[i,j]
      return self.R_s[s,t_past,i,j]
    else:

      return 0


  def create_result(self):

    self.Z = {}
    for stj in self.STJ:
      self.Z[stj] = 0

    self.A = {}
    for stj in self.STJ:
      self.A[stj] = 0

    self.XY = {}
    for sj in self.SJ:
      self.XY[sj] = 0

    self.Z_i = {}
    for j in self.D:
      self.Z_i[j] = self.Z2I[j]

    for s in self.S:

      for j in self.D:
        self.Z[s,1,j] = self.Z_i[j]
      
      #flow conservation
      for t in self.T:
        for j in self.D:
          if t in self.T2reb:
            self.Z[s,t+1,j] = self.Z[s,t,j] - sum(self.s_trip[s,t,j,i] for i in self.D) - sum(self.R_s[s,t,j,i] for i in self.D) + sum(self.e_trip[s,t,i,j] for i in self.D)
          else:
            self.Z[s,t+1,j] = self.Z[s,t,j] - sum(self.s_trip[s,t,j,i] for i in self.D) + sum(self.e_trip[s,t,i,j] for i in self.D) + sum(self.func_R_s_arr(s,t,i,j) for i in self.D)

      #violations of station capacity
      for t in self.T_:
          for j in self.D:
            if self.Z[s,t,j] > self.C2Max[j]:
              self.A[s,t,j] = self.Z[s,t,j] - self.C2Max[j]
            elif self.Z[s,t,j] < 0:
              self.A[s,t,j] = (-1)*self.Z[s,t,j]

      #deviations from the initial condition
      for j in self.D:
        self.XY[s,j] = abs(self.Z_i[j]-self.Z[s,self.time_num+1,j])

    return


  def no_reb_test(self):

    print('-- noreb --')

    for stij in self.ST2rebIJ:
      self.R_s[stij] = 0

    return


  def offline_reb_test(self):

    print('-- offline --')

    for s in self.S:
      for t in self.T2reb:
        for i in self.D:
            for j in self.D:
              self.R_s[s,t,i,j] = round(self.r[t,i,j].value())

    return


  def lincon_reb_test(self):

    print('-- lincon --')

    self.culc_trip_dif()
    for s in self.S:
      for t in self.T2reb:
        for i in self.D:
          for j in self.D:
            if round(self.r[t,i,j].value() + self.W[t,i,j]*(self.trip_dif[s,t,i]/self.C2Max[i]-self.trip_dif[s,t,j]/self.C2Max[j])) >= 0:
              self.R_s[s,t,i,j] = round(self.r[t,i,j].value() + self.W[t,i,j]*(self.trip_dif[s,t,i]/self.C2Max[i]-self.trip_dif[s,t,j]/self.C2Max[j]))
              self.R_s[s,t,j,i] = 0
            else:
              self.R_s[s,t,j,i] = (-1)*round(self.r[t,i,j].value() + self.W[t,i,j]*(self.trip_dif[s,t,i]/self.C2Max[i]-self.trip_dif[s,t,j]/self.C2Max[j]))
              self.R_s[s,t,i,j] = 0
           
    return


  def create_R_test(self):

    self.r = pickle_load(os.path.join(INTERMEDIATE_TRAIN_PATH, f'r.pickle'))
    self.w = pickle_load(os.path.join(INTERMEDIATE_TRAIN_PATH, f'w.pickle'))
    self.R_s = {}

    self.W = {}
    for t in self.T2reb:
      for i in self.D:
        for j in self.D:
          if self.w[t,i,j].value() == None:
            self.W[t,i,j] = 0
          else:
            self.W[t,i,j] = self.w[t,i,j].value()

    if self.control_name == 'noreb':
      self.no_reb_test()
    elif self.control_name == 'offline':
      self.offline_reb_test()
    elif self.control_name =='lincon':
      self.lincon_reb_test()
    else:
      pass

    self.const_allocation()

    return


  def create_opt_sum(self):

    self.r_s_sum = sum(self.r_s[s,t,i,j].value() for s in self.S for t in self.T2reb for i in self.D for j in self.D)
    self.a_sum = sum(self.a[s,t,j].value() for s in self.S for t in self.T_ for j in self.D)
    self.xy_sum = sum(self.x[s,j].value()+self.y[s,j].value() for s in self.S for j in self.D)
    self.w_sum = sum(self.w[t,i,j].value() for t in self.T2reb for i in self.D for j in self.D if self.w[t,i,j].value() != None)
    print('opt_sums:',self.r_s_sum,self.a_sum,self.xy_sum,self.w_sum)

    return


  def create_sum(self):

    self.R_s_sum = sum((self.R_s[s,t,i,j]) for s in self.S for t in self.T2reb for i in self.D for j in self.D)
    self.A_sum = sum(self.A[s,t,j] for s in self.S for t in self.T_ for j in self.D)
    self.XY_sum = sum(self.XY[s,j] for s in self.S for j in self.D)

    self.R_s_ave = self.R_s_sum/len(self.S)
    self.A_ave = self.A_sum/len(self.S)
    self.XY_ave = self.XY_sum/len(self.S)

    self.Obj = self.R_s_ave + self.A_ave + self.XY_ave

    print('FR:',self.R_s_ave,'FA:',self.A_ave,'FE:',self.XY_ave,'Obj',self.Obj)

    return


  def write_opt_result(self):

    pickle_dump(self.r, os.path.join(INTERMEDIATE_TRAIN_PATH, f'r.pickle'))
    pickle_dump(self.w, os.path.join(INTERMEDIATE_TRAIN_PATH, f'w.pickle'))
    pickle_dump(self.z, os.path.join(INTERMEDIATE_TRAIN_PATH, f'z.pickle'))
    pickle_dump(self.a, os.path.join(INTERMEDIATE_TRAIN_PATH, f'a.pickle'))
    pickle_dump(self.r_s, os.path.join(INTERMEDIATE_TRAIN_PATH, f'r_s.pickle'))

    pickle_dump(self.r, os.path.join(INTERMEDIATE_TEST_PATH, f'r.pickle'))
    pickle_dump(self.w, os.path.join(INTERMEDIATE_TEST_PATH, f'w.pickle'))

    return


  def write_train_result(self):

    pickle_dump(self.R_s, os.path.join(INTERMEDIATE_TRAIN_PATH, f'R_s.pickle'))
    pickle_dump(self.A, os.path.join(INTERMEDIATE_TRAIN_PATH, f'A.pickle'))
    pickle_dump(self.XY, os.path.join(INTERMEDIATE_TRAIN_PATH, f'XY.pickle'))
    pickle_dump(self.Z, os.path.join(INTERMEDIATE_TRAIN_PATH, f'Z.pickle'))

    return


  def write_test_result(self):

    pickle_dump(self.R_s, os.path.join(INTERMEDIATE_TEST_PATH, f'r_s.pickle'))
    pickle_dump(self.A, os.path.join(INTERMEDIATE_TEST_PATH, f'a.pickle'))
    pickle_dump(self.XY, os.path.join(INTERMEDIATE_TEST_PATH, f'xy.pickle'))
    pickle_dump(self.Z, os.path.join(INTERMEDIATE_TEST_PATH, f'z.pickle'))

    return


  def postprocess_train(self):
    print('==== POSTPROCESS ====')

    print('--- Create R train ---')
    self.create_R_train()

    print('--- Create Result ---')
    self.create_result()

    print('--- Create sum train ---')
    self.create_sum()

    print('--- Create opt sum train ---')  
    self.create_opt_sum()

    print('--- Write opt train ---')
    self.write_opt_result()

    print('--- Write train result ---')
    self.write_train_result()

    return
    

  def postprocess_test(self):
    print('==== POSTPROCESS ====')

    self.start = time.time()

    print('--- Create R test ---')
    self.create_R_test()

    print('--- Create Result test ---')
    self.create_result()

    self.end = time.time()
    self.elapsed_time = (self.end - self.start)

    print('--- Create sum test ---')
    self.create_sum()

    print('--- Write test result ---')
    self.write_test_result()

    return


def write_summary(ms_train, ms_test):
  
  with open(os.path.join(OUTPUT_SUMMARY_PATH,'summary.txt'), 'w') as f:

    f.write('Train RelVeh    :%f\n' % ms_train.R_s_ave)
    f.write('Train StaCap    :%f\n' % ms_train.A_ave)
    f.write('Train IniCon    :%f\n' % ms_train.XY_ave)
    f.write('Train Obj       :%f\n\n' % ms_train.Obj)

    f.write('Test  RelVeh    :%f\n' % ms_test.R_s_ave)
    f.write('Test  StaCap    :%f\n' % ms_test.A_ave)
    f.write('Test  IniCon    :%f\n' % ms_test.XY_ave)
    f.write('Test  Obj       :%f\n\n' % ms_test.Obj)

    f.write('Train Elapsed Time  :%f\n' % ms_train.elapsed_time)
    f.write('Test  Elapsed Time  :%f\n\n' % ms_test.elapsed_time)

    f.write('Train W Sum         :%f\n' % ms_train.elapsed_time)
    f.write('Train Status Name   :{}'.format(ms_train.status_name))


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

  print('===== TRAIN =====')
  ms_train = MobilitySolver()
  ms_train.preprocess_train()
  ms_train.solve()
  ms_train.postprocess_train()

  print('===== TEST =====')
  ms_test = MobilitySolver()
  ms_test.preprocess_test()
  ms_test.postprocess_test()

  write_summary(ms_train, ms_test)

  return


if __name__ == '__main__':

  main()
