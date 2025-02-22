#!/bin/bash

echo 'Carrying out model simulations...'
python simulation_mains/analytic_calcs_2base.py ar6_17 1
python simulation_mains/analytic_calcs_3base.py ar6_17 1
python simulation_mains/analytic_calcs_4base.py ar6_17 1

echo 'Making figure...'
python tab_mains/quoted_numbers.py ar6_17
python tab_mains/tab3_prems.py ar6_17 1
python tab_mains/tab4_scores.py ar6_17
python tab_mains/tab5_crossingtimes.py ar6_17
