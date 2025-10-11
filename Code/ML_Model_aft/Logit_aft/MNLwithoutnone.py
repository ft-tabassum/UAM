########################################
#
# @file biogeme_uam_pilot_with_specifiedCO.py
# @author: Mengying Fu, Raoul Rothfeld
# @date: 30/01/2018
#
#######################################

from biogeme import *
from headers import *
from loglikelihood import *
from statistics import *

#Parameters to be estimated
# Arguments:
#   - 1  Name for report; Typically, the same as the variable.
#   - 2  Starting value.
#   - 3  Lower bound.
#   - 4  Upper bound.
#   - 5  0: estimate the parameter, 1: keep it fixed.
#
ASC_CAR = Beta('ASC_CAR',0,-1000,1000,0,'Car cte.')
ASC_PT = Beta('ASC_PT',0,-1000,1000,1,'Public transport cte.')
ASC_AT = Beta('ASC_AT',0,-1000,1000,0,'Autonomous taxi cte.')
ASC_AFT = Beta('ASC_AFT',0,-1000,1000,0,'Autonomous flying taxi cte.')

B_COST = Beta('B_COST',0,-1000,1000,0,'Travel cost')
B_TIME = Beta('B_TIME',0,-1000,1000,0,'Travel time')

B_CAR_TIME = Beta('B_CAR_TIME',0,-1000,1000,0,'Car travel time')
B_PT_TIME = Beta('B_PT_TIME',0,-1000,1000,0,'Public transport travel time')
B_AT_TIME = Beta('B_AT_TIME',0,-1000,1000,0,'Autonomous taxi travel time')
B_AFT_TIME = Beta('B_AFT_TIME',0,-1000,1000,0,'Autonomous flying taxi travel time')

B_CAR_COST = Beta('B_CAR_COST',0,-1000,1000,0,'Car travel cost')
B_PT_COST = Beta('B_PT_COST',0,-1000,1000,0,'Public transport travel cost')
B_AT_COST = Beta('B_AT_COST',0,-1000,1000,0,'Autonomous taxi travel cost')
B_AFT_COST = Beta('B_AFT_COST',0,-1000,1000,0,'Autonomous flying taxi travel cost')

B_CAR_VOT = Beta('B_CAR_VOT',0,-1000,1000,0,'Car value of time')
B_PT_VOT = Beta('B_PT_VOT',0,-1000,1000,0,'PT value of time')
B_AT_VOT = Beta('B_AT_VOT',0,-1000,1000,0,'AT value of time')
B_AFT_VOT = Beta('B_AFT_VOT',0,-1000,1000,0,'AFT value of time')

B_INC = Beta('B_INC',0,-1000,1000,0,'inconvenience indicated by total walking time and/or waiting time')
B_CAR_INC = Beta('B_CAR_INC',0,-1000,1000,0,'CAR inconvenience indicated by total walking time and/or waiting time')
B_PT_INC = Beta('B_PT_INC',0,-1000,1000,0,'PT inconvenience indicated by total walking time and/or waiting time')
B_AT_INC = Beta('B_AT_INC',0,-1000,1000,0,'AT inconvenience indicated by total walking time and/or waiting time')
B_AFT_INC = Beta('B_AFT_INC',0,-1000,1000,0,'FAT inconvenience indicated by total walking time and/or waiting time')



# Safety level
B_safer = Beta('B_safer',0,-1000,1000,0,'2 times safer than driving')
B_ds = Beta('B_ds',0,-1000,1000,1,'Driving level safety')
B_riskier = Beta('B_riskier',0,-1000,1000,0,'2 times riskier than driving')

B_safer_AT = Beta('B_safer_AT',0,-1000,1000,0,'AT 2 times safer than driving')
B_ds_AT = Beta('B_ds_AT',0,-1000,1000,1,'AT driving level safety')
B_riskier_AT = Beta('B_riskier_AT',0,-1000,1000,0,'AT 2 times riskier than driving')

B_safer_AFT = Beta('B_safer_AFT',0,-1000,1000,0,'AFT 2 times safer than driving')
B_ds_AFT = Beta('B_ds_AFT',0,-1000,1000,1,'AFT driving level safety')
B_riskier_AFT = Beta('B_riskier_AFT',0,-1000,1000,0,'AFT 2 times riskier than driving')



# Gender
B_MALE = Beta('B_MALE',0,-1000,1000,0,'Male')
B_FEMALE = Beta('B_FEMALE',0,-1000,1000,1,'female')

B_MALE_CAR = Beta('B_MALE_CAR',0,-1000,1000,0,'Male car')
B_FEMALE_CAR = Beta('B_FEMALE_CAR',0,-1000,1000,1,'female car')

B_MALE_PT = Beta('B_MALE_PT',0,-1000,1000,0,'Male PT')
B_FEMALE_PT = Beta('B_MALE_PT',0,-1000,1000,1,'Male PT')

B_MALE_AT = Beta('B_MALE_AT',0,-1000,1000,0,'Male AT')
B_FEMALE_AT = Beta('B_FEMALE_AT',0,-1000,1000,1,'Female AT')

B_MALE_AFT = Beta('B_MALE_AFT',0,-1000,1000,0,'Male AFT')
B_FEMALE_AFT = Beta('B_FEMALE_AFT',0,-1000,1000,1,'Female AFT')



# Age
B_AGE1 = Beta('B_AGE1',0,-1000,1000,0,'18-25')
B_AGE2 = Beta('B_AGE2',0,-1000,1000,0,'26-35')
B_AGE3 = Beta('B_AGE3',0,-1000,1000,1,'36-45')
B_AGE4 = Beta('B_AGE4',0,-1000,1000,0,'46-55')
B_AGE5 = Beta('B_AGE5',0,-1000,1000,0,'56-65')
B_AGE6 = Beta('B_AGE6',0,-1000,1000,0,'olderthan65')


B_AGE1_CAR = Beta('B_AGE1_CAR',0,-1000,1000,0,'AGE1 car')
B_AGE2_CAR = Beta('B_AGE2_CAR',0,-1000,1000,0,'AGE2 car')
B_AGE3_CAR = Beta('B_AGE3_CAR',0,-1000,1000,1,'AGE3 car')
B_AGE4_CAR = Beta('B_AGE4_CAR',0,-1000,1000,0,'AGE4 car')
B_AGE5_CAR = Beta('B_AGE5_CAR',0,-1000,1000,0,'AGE5 car')
B_AGE6_CAR = Beta('B_AGE6_CAR',0,-1000,1000,0,'AGE6 car')

B_AGE1_PT = Beta('B_AGE1_PT',0,-1000,1000,0,'AGE1 PT')
B_AGE2_PT = Beta('B_AGE2_PT',0,-1000,1000,0,'AGE2 PT')
B_AGE3_PT = Beta('B_AGE3_PT',0,-1000,1000,1,'AGE3 PT')
B_AGE4_PT = Beta('B_AGE4_PT',0,-1000,1000,0,'AGE4 PT')
B_AGE5_PT = Beta('B_AGE5_PT',0,-1000,1000,0,'AGE5 PT')
B_AGE6_PT = Beta('B_AGE6_PT',0,-1000,1000,0,'AGE6 PT')

B_AGE1_AT = Beta('B_AGE1_AT',0,-1000,1000,0,'AGE1 AT')
B_AGE2_AT = Beta('B_AGE2_AT',0,-1000,1000,0,'AGE2 AT')
B_AGE3_AT = Beta('B_AGE3_AT',0,-1000,1000,1,'AGE3 AT')
B_AGE4_AT = Beta('B_AGE4_AT',0,-1000,1000,0,'AGE4 AT')
B_AGE5_AT = Beta('B_AGE5_AT',0,-1000,1000,0,'AGE5 AT')
B_AGE6_AT = Beta('B_AGE6_AT',0,-1000,1000,0,'AGE6 AT')

B_AGE1_AFT = Beta('B_AGE1_AFT',0,-1000,1000,0,'AGE1 AFT')
B_AGE2_AFT = Beta('B_AGE2_AFT',0,-1000,1000,0,'AGE2 AFT')
B_AGE3_AFT = Beta('B_AGE3_AFT',0,-1000,1000,1,'AGE3 AFT')
B_AGE4_AFT = Beta('B_AGE4_AFT',0,-1000,1000,0,'AGE4 AFT')
B_AGE5_AFT = Beta('B_AGE5_AFT',0,-1000,1000,0,'AGE5 AFT')
B_AGE6_AFT = Beta('B_AGE6_AFT',0,-1000,1000,0,'AGE6 AFT')

# Age groups above 46 has been merged into one group called "older", and meanwhile, the regarding AT and AFT of this group has been merged into one group called "Older AUTO"
B_OLDER_AUTO = Beta('B_OLDER_AUTO',0,-1000,1000,0,'OLDERAGE AUTO')

# Preference of car and AT for age6 were merged
B_AGE6_MODES = Beta('B_AGE6_MODES',0,-1000,1000,0,'Older than 65 car and AFT')

# Education
B_LOWERTHANBSC = Beta('B_LOWERTHANBSC',0,-1000,1000,0,'Lower than bachelor')
B_BSC = Beta('B_BSC',0,-1000,1000,1,'Bachelor')
B_MSC = Beta('B_MSC',0,-1000,1000,0,'Master')
B_PHD = Beta('B_PHD',0,-1000,1000,0,'Phd')

B_LOWERTHANBSC_CAR = Beta('B_LOWERTHANBSC_CAR',0,-1000,1000,0,'Lower than bachelor Car')
B_BSC_CAR = Beta('B_BSC_CAR',0,-1000,1000,1,'Bachelor Car')
B_MSC_CAR = Beta('B_MSC_CAR',0,-1000,1000,0,'Master Car')
B_PHD_CAR = Beta('B_PHD_CAR',0,-1000,1000,0,'Phd Car')

B_LOWERTHANBSC_PT = Beta('B_LOWERTHANBSC_PT',0,-1000,1000,0,'Lower than bachelor PT')
B_BSC_PT = Beta('B_BSC_PT',0,-1000,1000,1,'Bachelor PT')
B_MSC_PT = Beta('B_MSC_PT',0,-1000,1000,0,'Master PT')
B_PHD_PT = Beta('B_PHD_PT',0,-1000,1000,0,'Phd PT')

B_LOWERTHANBSC_AT = Beta('B_LOWERTHANBSC_AT',0,-1000,1000,0,'Lower than bachelor AT')
B_BSC_AT = Beta('B_BSC_AT',0,-1000,1000,1,'Bachelor AT')
B_MSC_AT = Beta('B_MSC_AT',0,-1000,1000,0,'Master AT')
B_PHD_AT = Beta('B_PHD_AT',0,-1000,1000,0,'Phd AT')

B_LOWERTHANBSC_AFT = Beta('B_LOWERTHANBSC_AFT',0,-1000,1000,0,'Lower than bachelor AFT')
B_BSC_AFT = Beta('B_BSC_AFT',0,-1000,1000,1,'Bachelor AFT')
B_MSC_AFT = Beta('B_MSC_AFT',0,-1000,1000,0,'Master AFT')
B_PHD_AFT = Beta('B_PHD_AFT',0,-1000,1000,0,'Phd AFT')

# Children
B_CHILDREN = Beta('B_CHILDREN',0,-1000,1000,0,'Have children')
B_NOCHILDREN = Beta('B_NOCHILDREN',0,-1000,1000,1,'No children')

B_CHILDREN_CAR = Beta('B_CHILDREN_CAR',0,-1000,1000,0,'Have children car')
B_NOCHILDREN_CAR = Beta('B_NOCHILDREN_CAR',0,-1000,1000,1,'No children car')

B_CHILDREN_PT = Beta('B_CHILDREN_PT',0,-1000,1000,0,'Have children PT')
B_NOCHILDREN_PT = Beta('B_NOCHILDREN_PT',0,-1000,1000,1,'No children PT')

B_CHILDREN_AT = Beta('B_CHILDREN_AT',0,-1000,1000,0,'Have children AT')
B_NOCHILDREN_AT = Beta('B_NOCHILDREN_AT',0,-1000,1000,1,'No children AT')

B_CHILDREN_AFT = Beta('B_CHILDREN_AFT',0,-1000,1000,0,'Have children AFT')
B_NOCHILDREN_AFT = Beta('B_NOCHILDREN_AFT',0,-1000,1000,1,'No children AFT')

# Current mode
B_CARUSER = Beta('B_CARUSER',0,-1000,1000,1,'Car user')
B_PTUSER = Beta('B_PTUSER',0,-1000,1000,0,'PT user')
B_SMUSER = Beta('B_SMUSER',0,-1000,1000,0,'SM user')

B_CARUSER_CAR = Beta('B_CARUSER_CAR',0,-1000,1000,1,'Caruser car')
B_PTUSER_CAR = Beta('B_PTUSER_CAR',0,-1000,1000,0,'PT user car')
B_SMUSER_CAR = Beta('B_SMUSER_CAR',0,-1000,1000,0,'SM user car')

B_CARUSER_PT = Beta('B_CARUSER_PT',0,-1000,1000,1,'Car user PT')
B_PTUSER_PT = Beta('B_PTUSER_PT',0,-1000,1000,0,'PT user PT')
B_SMUSER_PT = Beta('B_SMUSER_PT',0,-1000,1000,0,'SM user PT')

B_CARUSER_AT = Beta('B_CARUSER_AT',0,-1000,1000,1,'Car user AT')
B_PTUSER_AT = Beta('B_PTUSER_AT',0,-1000,1000,0,'PT user AT')
B_SMUSER_AT = Beta('B_SMUSER_AT',0,-1000,1000,0,'SM user AT')

B_CARUSER_AFT = Beta('B_CARUSER_AFT',0,-1000,1000,1,'Car user AFT')
B_PTUSER_AFT = Beta('B_PTUSER_AFT',0,-1000,1000,0,'PT user AFT')
B_SMUSER_AFT = Beta('B_SMUSER_AFT',0,-1000,1000,0,'SM user AFT')

B_CARUSER_NONE = Beta('B_CARUSER_NONE',0,-1000,1000,1,'Car user NONE')
B_PTUSER_NONE = Beta('B_PTUSER_NONE',0,-1000,1000,0,'PT user NONE')
B_SMUSER_NONE = Beta('B_SMUSER_NONE',0,-1000,1000,0,'SM user NONE')

# PT users' preference regarding AT and AFT has been merged into one group called "PT user auto"
B_PTUSER_AUTO = Beta('B_PTUSER_AUTO',0,-1000,1000,0,'PT user autonomous modes')

# Car availability
B_HAVECAR = Beta('B_HAVECAR',0,-1000,1000,1,'Have car')
B_NOCAR = Beta('B_NOCAR',0,-1000,1000,0,'No car')

B_HAVECAR_CAR = Beta('B_HAVECAR_CAR',0,-1000,1000,1,'Have car car')
B_NOCAR_CAR = Beta('B_NOCAR_CAR',0,-1000,1000,0,'No car car')

B_HAVECAR_PT = Beta('B_HAVECAR_PT',0,-1000,1000,1,'Have car PT')
B_NOCAR_PT = Beta('B_NOCAR_PT',0,-1000,1000,0,'No car PT')

B_HAVECAR_AT = Beta('B_CAR_AT',0,-1000,1000,1,'Have car AT')
B_NOCAR_AT = Beta('B_NOCAR_AT',0,-1000,1000,0,'No car AT')

B_HAVECAR_AFT = Beta('B_CAR_AFT',0,-1000,1000,1,'Have car AFT')
B_NOCAR_AFT = Beta('B_NOCAR_AFT',0,-1000,1000,0,'No car AFT')


# Employment
B_WORKING = Beta('B_WORKING',0,-1000,1000,1,'Working people')
B_STUDENT = Beta('B_STUDENT',0,-1000,1000,0,'Student')
B_OTHERS = Beta('B_OTHERS',0,-1000,1000,0,'Others')

B_WORKING_CAR = Beta('B_WORKING_CAR',0,-1000,1000,1,'Working people car')
B_STUDENT_CAR = Beta('B_STUDENT_CAR',0,-1000,1000,0,'Student car')
B_OTHERS_CAR = Beta('B_OTHERS_CAR',0,-1000,1000,0,'Others car')

B_WORKING_PT = Beta('B_WORKING_PT',0,-1000,1000,1,'Working people PT')
B_STUDENT_PT = Beta('B_STUDENT_PT',0,-1000,1000,0,'Student PT')
B_OTHERS_PT = Beta('B_OTHERS_PT',0,-1000,1000,0,'Others PT')

B_WORKING_AT = Beta('B_WORKING_AT',0,-1000,1000,1,'Working people AT')
B_STUDENT_AT = Beta('B_STUDENT_AT',0,-1000,1000,0,'Student AT')
B_OTHERS_AT = Beta('B_OTHERS_AT',0,-1000,1000,0,'Others AT')

B_WORKING_AFT = Beta('B_WORKING AFT',0,-1000,1000,1,'Working people AFT')
B_STUDENT_AFT = Beta('B_STUDENT AFT',0,-1000,1000,0,'Student AFT')
B_OTHERS_AFT = Beta('B_OTHERS AFT',0,-1000,1000,0,'Others AFT')

# Trip purpose
B_COM = Beta('B_COM',0,-1000,1000,0,'For commuting purpose')
B_NONCOM = Beta('B_NONCOM',0,-1000,1000,1,'For noncommuting purpose')

B_COM_CAR = Beta('B_COM_CAR',0,-1000,1000,0,'For commuting purpose car')
B_NONCOM_CAR = Beta('B_NONCOM_CAR',0,-1000,1000,1,'For noncommuting purpose car')

B_COM_PT = Beta('B_COM_PT',0,-1000,1000,0,'For commuting purpose PT')
B_NONCOM_PT = Beta('B_NONCOM_PT',0,-1000,1000,1,'For noncommuting purpose PT')

B_COM_AT = Beta('B_COM_AT',0,-1000,1000,0,'For commuting purpose AT')
B_NONCOM_AT = Beta('B_NONCOM_AT',0,-1000,1000,1,'For noncommuting purpose AT')

B_COM_AFT = Beta('B_COM_AFT',0,-1000,1000,0,'For commuting purpose AFT')
B_NONCOM_AFT = Beta('B_NONCOM_AFT',0,-1000,1000,1,'For noncommuting purpose AFT')

# Income
B_INCOME1 = Beta('B_INCOME1',0,-1000,1000,0,'less than 500')
B_INCOME2 = Beta('B_INCOME2',0,-1000,1000,0,'500-1000')
B_INCOME3 = Beta('B_INCOME3',0,-1000,1000,0,'1000-2000')
B_INCOME4 = Beta('B_INCOME4',0,-1000,1000,0,'2000-3000')
B_INCOME5 = Beta('B_INCOME5',0,-1000,1000,1,'3000-4000')
B_INCOME6 = Beta('B_INCOME6',0,-1000,1000,0,'4000-5000')
B_INCOME7 = Beta('B_INCOME7',0,-1000,1000,0,'5000-6000')
B_INCOME8 = Beta('B_INCOME8',0,-1000,1000,0,'6000-7000')
B_INCOME9 = Beta('B_INCOME9',0,-1000,1000,0,'more than 7000')


B_INCOME1_CAR = Beta('B_INCOME1_CAR',0,-1000,1000,0,'less than 500 CAR')
B_INCOME2_CAR = Beta('B_INCOME2_CAR',0,-1000,1000,0,'500-1000 CAR')
B_INCOME3_CAR = Beta('B_INCOME3_CAR',0,-1000,1000,0,'1000-2000 CAR')
B_INCOME4_CAR = Beta('B_INCOME4_CAR',0,-1000,1000,0,'2000-3000 CAR')
B_INCOME5_CAR = Beta('B_INCOME5_CAR',0,-1000,1000,1,'3000-4000 CAR')
B_INCOME6_CAR = Beta('B_INCOME6_CAR',0,-1000,1000,0,'4000-5000 CAR')
B_INCOME7_CAR = Beta('B_INCOME7_CAR',0,-1000,1000,0,'5000-6000 CAR')
B_INCOME8_CAR = Beta('B_INCOME8_CAR',0,-1000,1000,0,'6000-7000 CAR')
B_INCOME9_CAR = Beta('B_INCOME9_CAR',0,-1000,1000,0,'more than 7000 CAR')


B_INCOME1_PT = Beta('B_INCOME1_PT',0,-1000,1000,0,'less than 500 PT')
B_INCOME2_PT = Beta('B_INCOME2_PT',0,-1000,1000,0,'500-1000 PT')
B_INCOME3_PT = Beta('B_INCOME3_PT',0,-1000,1000,0,'1000-2000 PT')
B_INCOME4_PT = Beta('B_INCOME4_PT',0,-1000,1000,0,'2000-3000 PT')
B_INCOME5_PT = Beta('B_INCOME5_PT',0,-1000,1000,1,'3000-4000 PT')
B_INCOME6_PT = Beta('B_INCOME6_PT',0,-1000,1000,0,'4000-5000 PT')
B_INCOME7_PT = Beta('B_INCOME7_PT',0,-1000,1000,0,'5000-6000 PT')
B_INCOME8_PT = Beta('B_INCOME8_PT',0,-1000,1000,0,'6000-7000 PT')
B_INCOME9_PT = Beta('B_INCOME9_PT',0,-1000,1000,0,'more than 7000 PT')


B_INCOME1_AT = Beta('B_INCOME1_AT',0,-1000,1000,0,'less than 500 AT')
B_INCOME2_AT = Beta('B_INCOME2_AT',0,-1000,1000,0,'500-1000 AT')
B_INCOME3_AT = Beta('B_INCOME3_AT',0,-1000,1000,0,'1000-2000 AT')
B_INCOME4_AT = Beta('B_INCOME4_AT',0,-1000,1000,0,'2000-3000 AT')
B_INCOME5_AT = Beta('B_INCOME5_AT',0,-1000,1000,1,'3000-4000 AT')
B_INCOME6_AT = Beta('B_INCOME6_AT',0,-1000,1000,0,'4000-5000 AT')
B_INCOME7_AT = Beta('B_INCOME7_AT',0,-1000,1000,0,'5000-6000 AT')
B_INCOME8_AT = Beta('B_INCOME8_AT',0,-1000,1000,0,'6000-7000 AT')
B_INCOME9_AT = Beta('B_INCOME9_AT',0,-1000,1000,0,'more than 7000 AT')


B_INCOME1_AFT = Beta('B_INCOME1_AFT',0,-1000,1000,0,'less than 500 AFT')
B_INCOME2_AFT = Beta('B_INCOME2_AFT',0,-1000,1000,0,'500-1000 AFT')
B_INCOME3_AFT = Beta('B_INCOME3_AFT',0,-1000,1000,0,'1000-2000 AFT')
B_INCOME4_AFT = Beta('B_INCOME4_AFT',0,-1000,1000,0,'2000-3000 AFT')
B_INCOME5_AFT = Beta('B_INCOME5_AFT',0,-1000,1000,1,'3000-4000 AFT')
B_INCOME6_AFT = Beta('B_INCOME6_AFT',0,-1000,1000,0,'4000-5000 AFT')
B_INCOME7_AFT = Beta('B_INCOME7_AFT',0,-1000,1000,0,'5000-6000 AFT')
B_INCOME8_AFT = Beta('B_INCOME8_AFT',0,-1000,1000,0,'6000-7000 AFT')
B_INCOME9_AFT = Beta('B_INCOME9_AFT',0,-1000,1000,0,'more than 7000 AFT')




# Utility functions


# For numerical reasons, it is good practice to scale the data to
# that the values of the parameters are around 1.0. 
# A previous estimation with the unscaled data has generated
# parameters around -0.01 for both cost and time. Therefore, time and
# cost are multipled my 0.01.

# The following statements are designed to preprocess the data. It is
# like creating a new columns in the data file. This should be
# preferred to the statement like
# TRAIN_TT_SCALED = TRAIN_TT / 100.0
# which will cause the division to be reevaluated again and again,
# throuh the iterations. For models taking a long time to estimate, it
# may make a significant difference.

CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT/10)
CAR_COST_SCALED = DefineVariable('CAR_COST_SCALED', CAR_CO/10)
CAR_INC_SCALED = DefineVariable('CAR_INC_SCALED', CAR_INC/10)

PT_TT_SCALED = DefineVariable('PT_TT_SCALED', PT_TT/10)
PT_COST_SCALED = DefineVariable('PT_COST_SCALED', PT_CO/10)
PT_INC_SCALED = DefineVariable('PT_INC_SCALED', PT_INC/10)

AT_TT_SCALED = DefineVariable('AT_TT_SCALED', AT_TT/10)
AT_COST_SCALED = DefineVariable('AT_COST_SCALED', AT_CO/10)
AT_SF_riskier = DefineVariable('AT_SF_riskier', AT_SAFETY_riskier)
AT_SF_safer = DefineVariable('AT_SF_safer', AT_SAFETY_safer)
AT_INC_SCALED = DefineVariable('AT_INC_SCALED', AT_INC/10)

AFT_TT_SCALED = DefineVariable('AFT_TT_SCALED', AFT_TT/10)
AFT_COST_SCALED = DefineVariable('AFT_COST_SCALED', AFT_CO/10)
AFT_SF_riskier = DefineVariable('AFT_SF_riskier', AFT_SAFETY_riskier)
AFT_SF_safer = DefineVariable('AFT_SF_safer', AFT_SAFETY_safer)
AFT_INC_SCALED = DefineVariable('AFT_INC_SCALED', AFT_INC/10)



# socio.demographics

# Gender
MALE = DefineVariable('MALE', Gender == 1)
FEMALE = DefineVariable('FEMALE', Gender == 2)

# Age
AGE1 = DefineVariable('AGE1', Age == 2)
AGE2 = DefineVariable('AGE2', Age == 3)
AGE3 = DefineVariable('AGE3', Age == 4)
AGE4 = DefineVariable('AGE4', Age == 5)
AGE5 = DefineVariable('AGE5', Age == 6)
AGE6 = DefineVariable('AGE6', Age == 7)

# Education
LOWERTHANBSC = DefineVariable('LOWERTHANBSC', NewEducation == 1)
BSC = DefineVariable('BSC', NewEducation == 2)
MSC = DefineVariable('MSC', NewEducation == 3)
PHD = DefineVariable('PHD', NewEducation == 4)

# Children
CHILDREN = DefineVariable('CHILDREN', NewChildren == 1)
NOCHILDREN = DefineVariable('NOCHILDREN', NewChildren == 2)

# Current means of transport
CARUSER = DefineVariable('CARUSER', Newtransportmode == 1)
PTUSER = DefineVariable('PTUSER', Newtransportmode == 2)
SMUSER = DefineVariable('SMUSER', Newtransportmode == 3)

# Car availability
HAVECAR = DefineVariable('HAVECAR', NewCaravilability == 1)
NOCAR = DefineVariable('NOCAR', NewCaravilability == 2)

# Employment
WORKING = DefineVariable('WORKING', NewEmployment == 1)
STUDENT = DefineVariable('STUDENT', NewEmployment == 2)
OTHERS = DefineVariable('OTHERS', NewEmployment == 3)

# Trip purpose
COM = DefineVariable('COM', Commuting == 1)
NONCOM = DefineVariable('NONCOM', Commuting == 0)

# Income
INCOME1 = DefineVariable('INCOME1', Income == 1)
INCOME2 = DefineVariable('INCOME2', Income == 2)
INCOME3 = DefineVariable('INCOME3', Income == 3)
INCOME4 = DefineVariable('INCOME4', Income == 4)
INCOME5 = DefineVariable('INCOME5', Income == 5)
INCOME6 = DefineVariable('INCOME6', Income == 6)
INCOME7 = DefineVariable('INCOME7', Income == 7)
INCOME8 = DefineVariable('INCOME8', Income == 8)
INCOME9 = DefineVariable('INCOME9', Income == 9)


V1 = B_CAR_TIME * CAR_TT_SCALED + B_CAR_COST * CAR_COST_SCALED + B_AGE3_CAR * AGE3 + B_AGE5_CAR * AGE5 + B_AGE6_MODES * AGE6 + B_BSC_CAR * BSC + B_CARUSER_CAR * CARUSER + B_PTUSER_CAR * PTUSER + B_SMUSER_CAR * SMUSER + B_HAVECAR_CAR * HAVECAR + B_NOCAR_CAR * NOCAR + B_WORKING_CAR * WORKING + B_STUDENT_CAR * STUDENT + B_OTHERS_CAR * OTHERS + B_COM_CAR * COM + B_NONCOM_CAR * NONCOM + B_INCOME1_CAR * INCOME1 + B_INCOME5_CAR * INCOME5 + B_INCOME9_CAR * INCOME9 
V2 = ASC_PT + B_PT_TIME * PT_TT_SCALED + B_PT_COST * PT_COST_SCALED + B_PT_INC * PT_INC_SCALED
V3 = ASC_AT + B_AT_TIME * AT_TT_SCALED + B_AT_COST * AT_COST_SCALED + B_safer_AT * AT_SAFETY_safer + B_riskier_AT * AT_SAFETY_riskier + B_AGE3_AT * AGE3 + B_OLDER_AUTO * AGE4 + B_OLDER_AUTO * AGE5 + B_BSC_AT * BSC + B_CARUSER_AT * CARUSER + B_PTUSER_AUTO * PTUSER + B_SMUSER_AT * SMUSER + B_HAVECAR_AT * HAVECAR + B_NOCAR_AT * NOCAR + B_WORKING_AT * WORKING + B_COM_AT * COM + B_NONCOM_AT * NONCOM + B_INCOME2_AT * INCOME2 + B_INCOME5_AT * INCOME5 + B_INCOME9_AT * INCOME9
V4 = ASC_AFT + B_AFT_TIME * AFT_TT_SCALED + B_AFT_COST * AFT_COST_SCALED + B_riskier_AFT * AFT_SAFETY_riskier + B_AGE3_AFT * AGE3 + B_OLDER_AUTO * AGE4 + B_OLDER_AUTO * AGE5 + B_AGE6_MODES * AGE6 + B_BSC_AFT * BSC + B_CARUSER_AFT * CARUSER + B_PTUSER_AUTO * PTUSER + B_SMUSER_AFT * SMUSER + B_HAVECAR_AFT * HAVECAR + B_NOCAR_AFT * NOCAR + B_WORKING_AFT * WORKING + B_STUDENT_AFT * STUDENT + B_COM_AFT * COM + B_NONCOM_AFT * NONCOM + B_INCOME2_AFT * INCOME2 + B_INCOME3_AFT * INCOME3 + B_INCOME4_AFT * INCOME4 + B_INCOME5_AFT * INCOME5 + B_INCOME8_AFT * INCOME8 + B_INCOME9_AFT * INCOME9 + B_NOCHILDREN_AFT * NOCHILDREN

# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2,
     3: V3,
     4: V4}

# Associate the availability condition to the alternatives
av = {1: 1,
      2: 1,
      3: 1,
      4: 1}

# The choice model is a logit
logprob = bioLogLogit(V,av,CHOICE)

# Defines an itertor on the data
rowIterator('obsIter') 

# DEfine the likelihood function for the estimation
BIOGEME_OBJECT.ESTIMATE = Sum(logprob,'obsIter')

# All observations verifying the following expression will not be
# considered for estimation 
# Observations such that the dependent variable CHOICE is 0 are removed.

BIOGEME_OBJECT.EXCLUDE = (CHOICE == 0)

# Statistics

nullLoglikelihood(av,'obsIter')
choiceSet = [1,2,3,4]
cteLoglikelihood(choiceSet,CHOICE,'obsIter')
availabilityStatistics(av,'obsIter')


BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = "BIO"

BIOGEME_OBJECT.FORMULAS['Car utility'] = V1
BIOGEME_OBJECT.FORMULAS['Public transport utility'] = V2
BIOGEME_OBJECT.FORMULAS['Autonomous taxi utility'] = V3
BIOGEME_OBJECT.FORMULAS['Autonomous flying taxi utility'] = V4






