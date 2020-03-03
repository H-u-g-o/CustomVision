import os, collections
from secret_exakis import *
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import image, PredictionQueryToken, Iteration


class customVisionApi:
    '''Methods to call Custom Vision Prediction & Training API.'''
    def __init__(self, endpoint=ENDPOINT, training_key=training_key, prediction_key=prediction_key, prediction_resource_id=prediction_ressource_id, projectId=projectId_PL, Iteration_id=Iteration_id_PL, publish_iteration_name=publish_iteration_name_PL, training_images=playing_set):
     self.endpoint=endpoint
     self.training_key=training_key
     self.prediction_key=prediction_key
     self.prediction_resource_id=prediction_resource_id
     self.projectId=projectId
     self.Iteration_id=Iteration_id
     self.publish_iteration_name=publish_iteration_name
     self.training_images=training_images
    #Initializing Prediction Client
     self.predictor = CustomVisionPredictionClient(self.prediction_key, self.endpoint)
    #Initializing Training Client
     self.trainer = CustomVisionTrainingClient(self.training_key, self.endpoint)

    def predictNoStore(self):
        '''Browse a local file & apply Custom Vision Detection on each image without storing them. Results are returned only if a default has come up with a probability of 10% or more.'''
        #List all images in directory
        directory = os.listdir(self.training_images)
        print("Found:", len(directory), "images")

        listAllDefaultsInFile = []

        #Apply prediction on every image
        for file in directory:
            with open(self.training_images+'/'+file, "rb") as image_contents:
                results = self.predictor.detect_image_with_no_store(self.projectId, self.publish_iteration_name, image_contents.read())
                allDefault = {}
                fileName = file
                #List all defaults in image with probability
                getDefaultList = self.getDefaults(results.predictions)
                #Create default categories with sum of probability
                countDefault = self.defaultCounter(getDefaultList)
                #Return Tag Name with probability > 10%
                getSelectedTag = self.returnDefaultTag(countDefault)
                #Create a list of dict with all defaults found in the file images
                allDefault[fileName] = getSelectedTag
                print(allDefault)
                #Add default to list
                listAllDefaultsInFile.append(dict(allDefault))

        #Write result in file with name of the itteration
        saveResults = self.writeResult(listAllDefaultsInFile)


    def writeResult(self, listToBeWritten):
        '''Create a .txt file with the iteration name & write list content.'''
        fileName = 'result_'+self.publish_iteration_name+'.txt'
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
        #Create batches of 64 images (maximum for delete) & delete iamges
        batch = [allIds[i: i+64] for i in range(0, len(allIds), 64)]
        for i in batch:
            delete = self.trainer.delete_prediction(self.projectId, ids=i)
        return print("All images deleted")


if __name__ == "__main__":
    #p = customVisionApi()
    #p.predictNoStore()