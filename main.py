
def start():
    print("1. Insert Dataset")
    print("2. Choose What to predict")
    
    data_ingestion = Data_Ingestion()
    data_validation = Data_Validation()
    data_tranformation = Data_Transformation()
    model_trainer = Model_Trainer()
    model_evaluation = Model_Evaluation()
    
    data_ingestion.initiate_data_ingestion()
    
    choice = input("What do you want to predict?\n")
    
    data_validation.initiate_data_validation()
    data_tranformation.initiate_data_transformation(choice)
    model_trainer.initiate_model_training()
    model_evaluation.initiate_model_evaluation()