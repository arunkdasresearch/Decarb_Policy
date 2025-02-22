#!/bin/bash

echo 'Carrying out model simulations...'
python simulation_mains/analytic_calcs_2base.py ar6_17 1
python simulation_mains/analytic_calcs_3base.py ar6_17 1
python simulation_mains/analytic_calcs_4base.py ar6_17 1

echo 'Making figures...'
echo 'Figure 1...'
python figure_mains/pfig1_decarbdates_val.py ar6_17 1 

echo 'Figure 2...'
python figure_mains/pfig2_carbonprices.py ar6_17 1

echo 'Figure 3...'
python figure_mains/pfig3_eol.py ar6_17 1

echo 'Figure 4...'
python figure_mains/pfig4_paths.py ar6_17 1

echo 'Figure 5...'
python figure_mains/pfig5_seccostinds.py ar6_17 1

echo 'Figure 6...'
python figure_mains/pfig6_aggcost.py ar6_17 1

echo 'Figure 7...'
python figure_mains/pfig7_secatt_costs.py ar6_17 1

echo 'Figure 8...'
python figure_mains/pfig8_emis_temp.py ar6_17 1

echo 'Computed the quoted numbers and tables...'
python tab_mains/quoted_numbers.py ar6_17
python tab_mains/tab3_prems.py ar6_17 1
python tab_mains/tab4_scores.py ar6_17 1
python tab_mains/tab5_crossingtimes.py ar6_17
