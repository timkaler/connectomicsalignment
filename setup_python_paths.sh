#!/bin/bash
#source /afs/csail/proj/courses/6.172/scripts/.bashrc_silent
#source /afs/csail/proj/courses/6.172/scripts/.bashrc_silent
export ALIGNER=/efs/home/tfk/rh_aligner 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/:/usr/local/lib:/efs/tfk/.opencv-2.4.6.1/lib/ 
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efs/home/tfk/pythonroot/lib.linux-x86_64-2.7/ 
#export PYTHONPATH=:~/rh_aligner/dist:~/rh_aligner/:~/rh_logger/:~/rh_config/:$PYTHONPATH:/usr/lib/python2.7/:/usr/local/lib/python2.7/:/efs/tfk/.opencv-2.4.6.1/lib/python2.7/dist-packages/ 

#export PYTHONPATH=:~/rh_aligner/dist:~/rh_aligner/:~/rh_logger/:~/rh_config/:$PYTHONPATH:/usr/lib/python2.7/dist-packages:/usr/local/lib/python2.7/dist-packages:/efs/tfk/.opencv-2.4.6.1/lib/python2.7/dist-packages/:/usr/local/lib



#export PYTHONPATH=/efs/home/tfk/pythonroot/lib.linux-x86_64-2.7/
#export PYTHONHOME=/efs/home/tfk/pythonroot/usr/local/lib/:/usr/local/lib

export PYTHONHOME=/efs/home/tfk/pythonroot/usr
#export PYTHONPATH=/usr/lib/python2.7:/usr/local/lib/python2.7:/usr/local/lib/python2.7/dist-packages:/usr/local/lib/python2.7/site-packages
export PYTHONPATH=/efs/tfk/.opencv-2.4.6.1/lib/python2.7/dist-packages/:/efs/home/tfk/pythonroot/usr/local/lib/python2.7/dist-packages:/usr/local/lib/python2.7/dist-packages/backports_abc-0.5-py2.7.egg:/usr/local/lib/python2.7/dist-packages/certifi-2016.09.26-py2.7.egg:/usr/local/lib/python2.7/dist-packages/singledispatch-3.4.0.3-py2.7.egg:/usr/local/lib/python2.7/dist-packages/backports.ssl_match_hostname-3.5.0.1-py2.7.egg:/usr/local/lib/python2.7/dist-packages/mbeam-0.1-py2.7.egg:/usr/local/lib/python2.7/dist-packages/Pympler-0.4.3-py2.7.egg:/usr/local/lib/python2.7/dist-packages/lru_dict-1.1.6-py2.7-linux-x86_64.egg:/usr/lib/python2.7/dist-packages:/usr/lib/python2.7:/usr/lib/python2.7/plat-x86_64-linux-gnu:/usr/lib/python2.7/lib-tk:/usr/lib/python2.7/lib-old:/usr/lib/python2.7/lib-dynload:/usr/local/lib/python2.7/dist-packages

#python setup.py install --force --root /efs/home/tfk/testpythonroot/usr/local/lib/python2.7
#python2.7 -v
#python /efs/home/tfk/rh_aligner/scripts/wrappers/match_sift_features_and_filter_cv2.py 
#python /efs/home/tfk/rh_aligner/scripts/wrappers/match_sift_features_and_filter_cv2.py
#python -u /efs/home/tfk/rh_aligner/scripts/wrappers/match_sift_features_and_filter_cv2.py -o "test_box_temp2/matched_sifts/W01_Sec010/intra/21/intra_l10_21_outputs_lst.txt" -w 30 -c "conf/conf_example.json" -t 4 "/efs/ECS_test9_cropped/tilespecs/W01_Sec010.json" "test_box_temp2/matched_sifts/W01_Sec010/intra/21/intra_l10_21_features_lst1.txt" "test_box_temp2/matched_sifts/W01_Sec010/intra/21/intra_l10_21_features_lst2.txt" --index_pairs 21_1:21_2 21_1:21_3 21_1:21_4 21_1:21_5 21_1:21_6 21_1:21_7 21_2:21_3 21_2:21_7 21_2:21_8 21_2:21_9 21_2:21_19 21_3:21_4 21_3:21_9 21_3:21_10 21_3:21_11 21_4:21_5 21_4:21_11 21_4:21_12 21_4:21_13 21_5:21_6 21_5:21_13 21_5:21_14 21_5:21_15 21_6:21_7 21_6:21_15 21_6:21_16 21_6:21_17 21_7:21_17 21_7:21_18 21_7:21_19 21_8:21_9 21_8:21_19 21_8:21_20 21_8:21_21 21_8:21_37 21_9:21_10 21_9:21_21 21_9:21_22 21_10:21_11 21_10:21_22 21_10:21_23 21_10:21_24 21_11:21_12 21_11:21_24 21_11:21_25 21_12:21_13 21_12:21_25 21_12:21_26 21_12:21_27 21_13:21_14 21_13:21_27 21_13:21_28 21_14:21_15 21_14:21_28 21_14:21_29 21_14:21_30 21_15:21_16 21_15:21_30 21_15:21_31 21_16:21_17 21_16:21_31 21_16:21_32 21_16:21_33 21_17:21_18 21_17:21_33 21_17:21_34 21_18:21_19 21_18:21_34 21_18:21_35 21_18:21_36 21_19:21_36 21_19:21_37 21_20:21_21 21_20:21_37 21_20:21_38 21_20:21_39 21_20:21_61 21_21:21_22 21_21:21_39 21_21:21_40 21_22:21_23 21_22:21_40 21_22:21_41 21_23:21_24 21_23:21_41 21_23:21_42 21_23:21_43 21_24:21_25 21_24:21_43 21_24:21_44 21_25:21_26 21_25:21_44 21_25:21_45 21_26:21_27 21_26:21_45 21_26:21_46 21_26:21_47 21_27:21_28 21_27:21_47 21_27:21_48 21_28:21_29 21_28:21_48 21_28:21_49 21_29:21_30 21_29:21_49 21_29:21_50 21_29:21_51 21_30:21_31 21_30:21_51 21_30:21_52 21_31:21_32 21_31:21_52 21_31:21_53 21_32:21_33 21_32:21_53 21_32:21_54 21_32:21_55 21_33:21_34 21_33:21_55 21_33:21_56 21_34:21_35 21_34:21_56 21_34:21_57 21_35:21_36 21_35:21_57 21_35:21_58 21_35:21_59 21_36:21_37 21_36:21_59 21_36:21_60 21_37:21_60 21_37:21_61 21_38:21_39 21_38:21_61 21_39:21_40 21_40:21_41 21_41:21_42 21_42:21_43 21_43:21_44 21_44:21_45 21_45:21_46 21_46:21_47 21_47:21_48 21_48:21_49 21_49:21_50 21_50:21_51 21_51:21_52 21_52:21_53 21_53:21_54 21_54:21_55 21_55:21_56 21_56:21_57 21_57:21_58 21_58:21_59 21_59:21_60 21_60:21_61

$@
