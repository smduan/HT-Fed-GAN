# todo: change here
DATASET_NAME = 'adult'

DATASET_DCOL = {
    'adult': [
        'workclass',
        'education',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'gender',
        'native_country',
        'income_bracket'
    ],
    'clinical': [
        'anaemia',
        'diabetes',
        'high_blood_pressure',
        'sex',
        'smoking',
        'DEATH_EVENT'
    ],
    'covtype': [
        'Wilderness_Area_0',
        'Wilderness_Area_1',
        'Wilderness_Area_2',
        'Wilderness_Area_3',
        'Soil_Type0',
        'Soil_Type1',
        'Soil_Type2',
        'Soil_Type3',
        'Soil_Type4',
        'Soil_Type5',
        'Soil_Type6',
        'Soil_Type7',
        'Soil_Type8',
        'Soil_Type9',
        'Soil_Type10',
        'Soil_Type11',
        'Soil_Type12',
        'Soil_Type13',
        'Soil_Type14',
        'Soil_Type15',
        'Soil_Type16',
        'Soil_Type17',
        'Soil_Type18',
        'Soil_Type19',
        'Soil_Type20',
        'Soil_Type21',
        'Soil_Type22',
        'Soil_Type23',
        'Soil_Type24',
        'Soil_Type25',
        'Soil_Type26',
        'Soil_Type27',
        'Soil_Type28',
        'Soil_Type29',
        'Soil_Type30',
        'Soil_Type31',
        'Soil_Type32',
        'Soil_Type33',
        'Soil_Type34',
        'Soil_Type35',
        'Soil_Type36',
        'Soil_Type37',
        'Soil_Type38',
        'Soil_Type39',
        'label'
    ],
    'credit': ['label'],
    'intrusion': [
        'protocol_type',
        'service',
        'flag',
        'label'
    ]
}

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

COL_TYPE_ALL = {
    'adult': [
        ("age", CONTINUOUS),
        ("workclass", CATEGORICAL),
        ("fnlwgt", CONTINUOUS),
        ("education", ORDINAL,
         ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Prof-school",
          "Assoc-voc", "Assoc-acdm", "Some-college", "Bachelors", "Masters", "Doctorate"]),
        ("education_num", CONTINUOUS),
        ("marital_status", CATEGORICAL),
        ("occupation", CATEGORICAL),
        ("relationship", CATEGORICAL),
        ("race", CATEGORICAL),
        ("gender", CATEGORICAL),
        ("capital_gain", CONTINUOUS),
        ("capital_loss", CONTINUOUS),
        ("hours_per_week", CONTINUOUS),
        ("native_country", CATEGORICAL),
        ("income_bracket", CATEGORICAL)
    ],
    'clinical': [
        ("age", CONTINUOUS),
        ("anaemia", CATEGORICAL),
        ("creatinine_phosphokinase", CONTINUOUS),
        ("diabetes", CATEGORICAL),
        ("ejection_fraction", CONTINUOUS),
        ("high_blood_pressure", CATEGORICAL),
        ("platelets", CONTINUOUS),
        ("serum_creatinine", CONTINUOUS),
        ("serum_sodium", CONTINUOUS),
        ("sex", CATEGORICAL),
        ("smoking", CATEGORICAL),
        ("time", CONTINUOUS),
        ("DEATH_EVENT", CATEGORICAL)
    ],
    'covtype': [
        ("Elevation", CONTINUOUS),
        ("Aspect", CONTINUOUS),
        ("Slope", CONTINUOUS),
        ("Horizontal_Distance_To_Hydrology", CONTINUOUS),
        ("Vertical_Distance_To_Hydrology", CONTINUOUS),
        ("Horizontal_Distance_To_Roadways", CONTINUOUS),
        ("Hillshade_9am", CONTINUOUS),
        ("Hillshade_Noon", CONTINUOUS),
        ("Hillshade_3pm", CONTINUOUS),
        ("Horizontal_Distance_To_Fire_Points", CONTINUOUS),
        ("Wilderness_Area_0", CATEGORICAL),
        ("Wilderness_Area_1", CATEGORICAL),
        ("Wilderness_Area_2", CATEGORICAL),
        ("Wilderness_Area_3", CATEGORICAL),
        ("Soil_Type0", CATEGORICAL),
        ("Soil_Type1", CATEGORICAL),
        ("Soil_Type2", CATEGORICAL),
        ("Soil_Type3", CATEGORICAL),
        ("Soil_Type4", CATEGORICAL),
        ("Soil_Type5", CATEGORICAL),
        ("Soil_Type6", CATEGORICAL),
        ("Soil_Type7", CATEGORICAL),
        ("Soil_Type8", CATEGORICAL),
        ("Soil_Type9", CATEGORICAL),
        ("Soil_Type10", CATEGORICAL),
        ("Soil_Type11", CATEGORICAL),
        ("Soil_Type12", CATEGORICAL),
        ("Soil_Type13", CATEGORICAL),
        ("Soil_Type14", CATEGORICAL),
        ("Soil_Type15", CATEGORICAL),
        ("Soil_Type16", CATEGORICAL),
        ("Soil_Type17", CATEGORICAL),
        ("Soil_Type18", CATEGORICAL),
        ("Soil_Type19", CATEGORICAL),
        ("Soil_Type20", CATEGORICAL),
        ("Soil_Type21", CATEGORICAL),
        ("Soil_Type22", CATEGORICAL),
        ("Soil_Type23", CATEGORICAL),
        ("Soil_Type24", CATEGORICAL),
        ("Soil_Type25", CATEGORICAL),
        ("Soil_Type26", CATEGORICAL),
        ("Soil_Type27", CATEGORICAL),
        ("Soil_Type28", CATEGORICAL),
        ("Soil_Type29", CATEGORICAL),
        ("Soil_Type30", CATEGORICAL),
        ("Soil_Type31", CATEGORICAL),
        ("Soil_Type32", CATEGORICAL),
        ("Soil_Type33", CATEGORICAL),
        ("Soil_Type34", CATEGORICAL),
        ("Soil_Type35", CATEGORICAL),
        ("Soil_Type36", CATEGORICAL),
        ("Soil_Type37", CATEGORICAL),
        ("Soil_Type38", CATEGORICAL),
        ("Soil_Type39", CATEGORICAL),
        ("label", CATEGORICAL)
    ],
    'credit': [
        ("V0", CONTINUOUS),
        ("V1", CONTINUOUS),
        ("V2", CONTINUOUS),
        ("V3", CONTINUOUS),
        ("V4", CONTINUOUS),
        ("V5", CONTINUOUS),
        ("V6", CONTINUOUS),
        ("V7", CONTINUOUS),
        ("V8", CONTINUOUS),
        ("V9", CONTINUOUS),
        ("V10", CONTINUOUS),
        ("V11", CONTINUOUS),
        ("V12", CONTINUOUS),
        ("V13", CONTINUOUS),
        ("V14", CONTINUOUS),
        ("V15", CONTINUOUS),
        ("V16", CONTINUOUS),
        ("V17", CONTINUOUS),
        ("V18", CONTINUOUS),
        ("V19", CONTINUOUS),
        ("V20", CONTINUOUS),
        ("V21", CONTINUOUS),
        ("V22", CONTINUOUS),
        ("V23", CONTINUOUS),
        ("V24", CONTINUOUS),
        ("V25", CONTINUOUS),
        ("V26", CONTINUOUS),
        ("V27", CONTINUOUS),
        ("Amount", CONTINUOUS),
        ("label", CATEGORICAL)
    ],
    'intrusion': [
        ("duration", CONTINUOUS),
        ("protocol_type", CATEGORICAL),
        ("service", CATEGORICAL),
        ("flag", CATEGORICAL),
        ("src_bytes", CONTINUOUS),
        ("dst_bytes", CONTINUOUS),
        ("land", CONTINUOUS),
        ("wrong_fragment", CONTINUOUS),
        ("urgent", CONTINUOUS),
        ("hot", CONTINUOUS),
        ("num_failed_logins", CONTINUOUS),
        ("logged_in", CONTINUOUS),
        ("num_compromised", CONTINUOUS),
        ("root_shell", CONTINUOUS),
        ("su_attempted", CONTINUOUS),
        ("num_root", CONTINUOUS),
        ("num_file_creations", CONTINUOUS),
        ("num_shells", CONTINUOUS),
        ("num_access_files", CONTINUOUS),
        ("is_host_login", CONTINUOUS),
        ("is_guest_login", CONTINUOUS),
        ("count", CONTINUOUS),
        ("srv_count", CONTINUOUS),
        ("serror_rate", CONTINUOUS),
        ("srv_serror_rate", CONTINUOUS),
        ("rerror_rate", CONTINUOUS),
        ("srv_rerror_rate", CONTINUOUS),
        ("same_srv_rate", CONTINUOUS),
        ("diff_srv_rate", CONTINUOUS),
        ("srv_diff_host_rate", CONTINUOUS),
        ("dst_host_count", CONTINUOUS),
        ("dst_host_srv_count", CONTINUOUS),
        ("dst_host_same_srv_rate", CONTINUOUS),
        ("dst_host_diff_srv_rate", CONTINUOUS),
        ("dst_host_same_src_port_rate", CONTINUOUS),
        ("dst_host_srv_diff_host_rate", CONTINUOUS),
        ("dst_host_serror_rate", CONTINUOUS),
        ("dst_host_srv_serror_rate", CONTINUOUS),
        ("dst_host_rerror_rate", CONTINUOUS),
        ("dst_host_srv_rerror_rate", CONTINUOUS),
        ("label", CATEGORICAL)
    ]
}
