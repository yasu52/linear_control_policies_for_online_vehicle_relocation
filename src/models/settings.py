
input_trip_path    = 'input/trip_data_7'
input_station_path = 'input/station_data_7'
s_trip_train_file  = 's_trip_201507.pickle'
e_trip_train_file  = 'e_trip_201507.pickle'
s_trip_test_file   = 's_trip_201508.pickle'
e_trip_test_file   = 'e_trip_201508.pickle'

intermediate_train_path   = 'intermediate_data/mpc/201507_train_7'
intermediate_test_path    = 'intermediate_data/mpc/201508_test_7'

output_summary_path = 'output/summary/mpc/20150708_7'

station_num = 7
time_num = 288
reb_span = 1
reb_start = 8
reb_end = 20
R = 15
alpha = 1
beta = 1
gamma = 1
Tau_pol = 6
control_name = 'mpc_expect'