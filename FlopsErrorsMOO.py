from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.backend import clear_session
import optuna
from model_evaluator import ModelEvaluator
from model_trainer import ModelTrainer
import pandas as pd



# Erzeugen des ModelEvaluator zum Auswerten der FLOPs und des Trainers für den Train/Test_Split
model_evaluator = ModelEvaluator()
model_trainer = ModelTrainer()


# CNN Modell für den COIL-100 Datensatz. Dieses Modell soll automatisch optimiert werden.
def objective(trial):

    clear_session()

    X_train, X_test, y_train, y_test = model_trainer.get_train_test_cnn()
    input_shape = (128, 128, 3)
    
    numConvLayers = trial.suggest_int('numConvLayers', 1, 3)
 
    model = Sequential()
    for i in range(1, numConvLayers):
            model.add(Conv2D(filters = trial.suggest_categorical("filters"+str(i),[32,16]), 
                            kernel_size= trial.suggest_categorical("kernel_size"+str(i),[(3, 3),(5, 5)]), 
                            activation= trial.suggest_categorical("activation"+str(i), ['relu','linear']), 
                        input_shape=input_shape)
                ),
            model.add(MaxPooling2D(pool_size=(2, 2))),
    model.add(Dropout(trial.suggest_float("Dropout", 0.2 , 0.5))),
    model.add(Flatten()),
    model.add(Dense(128, activation='relu')),
    model.add(Dense(100, activation='softmax'))
    
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    opt = Adam(learning_rate=lr)  
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

    print(f"Training model ...")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)  
    BS = trial.suggest_categorical("BatchSize", [16, 32, 64, 128, 256] )
    
    model.fit(X_train, y_train, 
              epochs=5,
              batch_size = BS,
              steps_per_epoch=len(X_train) // BS,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping])
    
    print(f"Evaluating model ...")
    
    flops = model_evaluator.get_flops(model)
    acc = model.evaluate(X_test, y_test, verbose=1)[1]

    return acc, flops


#------------------------------------------------------------------------------------
#                               Optuna Studie
#------------------------------------------------------------------------------------
# Die Werte, die von objective(trial) zurück gegeben werden, sollen minimiert werden 
# => Error und FLOPs Anzahl soll minimiert werden.
# Mittels n_trials gibt man die Anzahl der Optimierungsversuche an.
study_name = "MultiObjectiveCNN-Example"
study = optuna.create_study(study_name = study_name, directions = ["maximize", "minimize"])
study.optimize(objective, n_trials=20)

#------------------------------------------------------------------------------------
#                               Auswertung
#------------------------------------------------------------------------------------
# Speichern aller Trials als csv-Datei
df = study.trials_dataframe()
df.to_excel('study_results.xlsx')

print(f"\n\nNumber of trials on the Pareto front: {len(study.best_trials)}")

trial_with_lowest_error = max(study.best_trials, key=lambda t: t.values[1])
print(f"\nTrial with highest Accuracy Rate: ")
print(f"\tnumber: {trial_with_lowest_error.number}")
print(f"\tparams: {trial_with_lowest_error.params}")
print(f"\tvalues: {trial_with_lowest_error.values}\n")

# Plotte Pareto-Front, sowie die Parameter mit dem größten Einfluss auf die einzelnen Optimierungsziele
fig1 = optuna.visualization.plot_pareto_front(study, target_names = ["Accuracy", "FLOPs"])
print("Plotte Fig1")
fig1.show(renderer = 'browser')
fig2 = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name = 'FLOPs')
print("Plotte Fig2")
fig2.show(renderer = 'browser')