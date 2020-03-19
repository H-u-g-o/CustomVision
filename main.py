import os, collections, argparse
from secret import *
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import image, PredictionQueryToken, Iteration


class customVisionApi:
    '''Methods to call Custom Vision Prediction & Training API.'''
    def __init__(self,
     endpoint,
     training_key,
     prediction_key,
     prediction_ressource_id,
     project_id,
     iteration_id,
     iteration_name,
     training_images,
  ):

     self.endpoint=endpoint
     self.training_key=training_key
     self.prediction_key=prediction_key
     self.prediction_ressource_id=prediction_ressource_id
     self.project_id=project_id
     self.iteration_id=iteration_id
     self.iteration_name=iteration_name
     self.training_images=training_images

    #Initializing Prediction Client
     self.predictor = CustomVisionPredictionClient(self.prediction_key, self.endpoint)

    #Initializing Training Client
     self.trainer = CustomVisionTrainingClient(self.training_key, self.endpoint)

    def listImagesInFile(self):
        '''Browse all images in a directory and split them in a dictionnary according if model name is present in file name'''
        model_name ="PL"
        images = {'Model_A':[],'Model_B':[]}
        #List all images in directory & display count
        directory = os.listdir(self.training_images)
        print("Found:", len(directory), "images")
        #Split images in dictionnary according to their name
        for image in directory :
            if model_name in image :
                images['Model_A'].append(image)
            else:
                images['Model_B'].append(image)
        print("Found:", len(images['Model_A']), "Model_A")
        print("Found:", len(images['Model_B']), "Model_B")
        return images

    def predictModelNoStore(self, imageToBeDetected):
        '''Apply Cusom Vision API on image and list defaults'''
        listAllDefaultsInImage = []
        with open(self.training_images+'/'+imageToBeDetected, "rb") as image_contents:
            results = self.predictor.detect_image_with_no_store(self.project_id, self.iteration_name, image_contents.read())
            allDefault = {}
            fileName = imageToBeDetected
            #List all defaults in image with probability
            getDefaultList = self.getDefaults(results.predictions)
            #Create default categories with sum of probability
            countDefault = self.defaultCounter(getDefaultList)
            #Return Tag Name with probability > 10%
            getSelectedTag = self.returnDefaultTag(countDefault)
            #Create a list of dict with all defaults found in the file images
            allDefault[fileName] = getSelectedTag
        return allDefault

    def writeResult(self, listToBeWritten):
        '''Create a .txt file with the iteration name & write list content.'''
        fileName = 'result_'+self.iteration_name+'.txt'
        with open(fileName,'a') as f:
            f.write(str(listToBeWritten))
            f.close()
        return print("Results saved in "+fileName)

    def getDefaults(self, predictionResults):
        '''List all default found in image with its probability.'''
        #Initialise empty list to store defaults
        default_list = []
        for prediction in predictionResults :
            #Initialize a dictionnary with key:Default & value:probability
            default_proba = {}
            default_proba[prediction.tag_name] = prediction.probability*100
            #Add default to list
            default_list.append(dict(default_proba))
        #Return list with all defaults and their probabilities
        return default_list

    def defaultCounter(self, ListDefaultsWithProb):
        '''Group default in categories and sum probabilities.'''
        #Initialise counter() from Collections to sum probabilities per default
        counter = collections.Counter()
        for probabilityDefault in ListDefaultsWithProb:
            counter.update(probabilityDefault)
        #Return dictionnary with default and sum probability
        defaultWithProbaSum = dict(counter)
        return defaultWithProbaSum

    def returnDefaultTag(self, defaultsDictionnary):
        "Return only default with a certain probability."
        #Return Dictionnary with only default tag where probability > 10%
        accepted_probability = 10
        OnlySelectedDefaultTag = {k for (k,v) in defaultsDictionnary.items() if v > accepted_probability}
        return OnlySelectedDefaultTag

    def predictQuick(self):
        '''Upload image and display predictions in terminal'''
        #List all images in directory
        directory = os.listdir(self.training_images)
        print ("Found:", len(directory), "images")
        #Upload image without storing, get prediction
        for file in directory:
            with open(self.training_images+'/'+file, "rb") as image_contents:
                self.results = self.predictor.detect_image(self.projectId, self.publish_iteration_name, image_contents.read())
                print("Image : " + file)
            # Display the results.
            for prediction in self.results.predictions:
                print("\t" + prediction.tag_name +": {0:.2f}%".format(prediction.probability * 100))

    def getIdsinPredictions(self):
        """List Ids of images present in prediction"""
        #Retrieve Prediction Token
        query = PredictionQueryToken(order_by='Newest', iteration_id=self.Iteration_id)
        #Get all predictions results
        response = self.trainer.query_predictions(self.projectId, query)
        #Create a list with all ids & return it
        list_ids = []
        for elt in response.results:
            list_ids.append(elt.id)
        print("Found:", len(list_ids), "images in prediction")
        return(list_ids)

    def deletePredictions(self):
        '''Delete images in prediction'''
        #List all predictions Id
        allIds = self.getIdsinPredictions()
        #Create batches of 64 images (maximum for delete) & delete images
        batch = [allIds[i: i+64] for i in range(0, len(allIds), 64)]
        for i in batch:
            delete = self.trainer.delete_prediction(self.projectId, ids=i)
        return print("All images deleted")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-e", "--endpoint", action="store", type=str, help="Endpoint", dest="endpoint", default="https://westeurope.api.cognitive.microsoft.com")
    arg_parser.add_argument("-t", "--training_key", action="store", type=str, help="Training-Key", dest="training_key", default=training_key)
    arg_parser.add_argument("-p", "--prediction_key", action="store", type=str, help="Prediction-Key", dest="prediction_key", default=prediction_key)
    arg_parser.add_argument("-r", "--prediction_ressource", action="store", type=str, help="Prediction Ressource Id", dest="prediction_ressource_id", default=prediction_ressource_id)
    arg_parser.add_argument("-A", "--project_A", action="store", type=str, help="project ID for model A", dest="project_id_A", default=projectId_PL)
    arg_parser.add_argument("-B", "--project_B", action="store", type=str, help="project ID for model B", dest="project_id_B", default=projectId_IR)
    arg_parser.add_argument("-nA", "--iteration_name_A", action="store", type=str, help="iteration name for model A", dest="iteration_name_A", default=publish_iteration_name_PL)
    arg_parser.add_argument("-nB", "--iteration_name_B", action="store", type=str, help="iteration name for model B", dest="iteration_name_B", default=publish_iteration_name_IR)
    arg_parser.add_argument("-iA", "--iteration_id_A", action="store", type=str, help="iteration id for model A", dest="iteration_id_A", default=Iteration_id_PL)
    arg_parser.add_argument("-iB", "--iteration_id_B", action="store", type=str, help="iteration id for model B", dest="iteration_id_B", default=Iteration_id_IR)
    arg_parser.add_argument("-f", "--file", action="store", type=str, help="Image file", dest="file", default=playing_set)
    args = arg_parser.parse_args()

    if (not args.endpoint or not args.training_key or not args.prediction_key or not args.prediction_ressource_id or not args.project_id_A or not args.project_id_B or not args.iteration_id_A or not args.iteration_id_B or not args.iteration_name_A or not args.iteration_name_B or not args.file):
        arg_parser.print_help()
        exit(-1)

    #Model object init for PL & IR
    pl = customVisionApi(
    endpoint=args.endpoint,
    training_key=args.training_key,
    prediction_key=args.prediction_key,
    prediction_ressource_id=args.prediction_ressource_id,
    project_id=args.project_id_A,
    iteration_id=args.iteration_id_A,
    iteration_name=args.iteration_name_A,
    training_images=args.file)

    ir = customVisionApi(
    endpoint=args.endpoint,
    training_key=args.training_key,
    prediction_key=args.prediction_key,
    prediction_ressource_id=args.prediction_ressource_id,
    project_id=args.project_id_B,
    iteration_id=args.iteration_id_B,
    iteration_name=args.iteration_name_B,
    training_images=args.file)

    #List all images in directory & display count
    directory = os.listdir(args.file)
    print("Found:", len(directory), "images")

    listAllDefaultsInImages = []

    #Split images according to their name & predict defaults
    for image in directory :
        if 'PL' in image :
            detect_defaults_PL = pl.predictModelNoStore(image)
            listAllDefaultsInImages.append(detect_defaults_PL)
            print(detect_defaults_PL)
        else:
            detect_defaults_IR = pl.predictModelNoStore(image)
            listAllDefaultsInImages.append(detect_defaults_IR)
            print(detect_defaults_IR)

    #Write predictions results in file
    with open('results.txt','a') as f:
            f.write(str(listAllDefaultsInImages))
            f.close()
    print("Results saved")