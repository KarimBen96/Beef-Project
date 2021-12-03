import mlflow
import mlflow.pytorch

def add_to_mlflow(model=None, batch_size=None, nb_epochs=None, learning_rate=None, optimizer=None, 
                  train_loss=None, test_loss=None, train_accuracy=None, test_accuracy=None
                  Project=None, Architecture=None, Images=None):
    """
    """

    with mlflow.start_run():
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('nb_epochs', nb_epochs )
        mlflow.log_param('optimizer', 'Adam')
        mlflow.log_param('learning_rate', learning_rate)
        
        mlflow.log_metric('train_loss', train_loss) 
        mlflow.log_metric('test_loss', test_loss)
        
        mlflow.log_metric('train_accuracy', train_accuracy) 
        mlflow.log_metric('test_accuracy', test_accuracy)
        
        mlflow.set_tag('Project', 'Beef')
        mlflow.set_tag('Architecture', 'CNN_3D')
        mlflow.set_tag('Images', '128-128-3')
        
        mlflow.pytorch.log_model(model, "model")
