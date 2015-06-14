import java.io.File;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Add;
import weka.core.converters.CSVSaver;
import weka.core.converters.ArffSaver;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.LibSVM;

import java.io.FileWriter;
import java.util.Date;
import java.util.Random;

public class test {
 
    public static double semisupervisedknn(Instances data, Instances datatest) throws Exception{
    	
    	Random rand = new Random();
    	double logloss = 0;
    	int length = data.numInstances();
    	while(datatest.numInstances() != 0)
    	{	//setup
    		 IBk knn = new IBk();
    		 knn.buildClassifier(data);
    		 int randomNum = rand.nextInt(datatest.numInstances());
    		 
    		 double score = knn.classifyInstance(datatest.instance(randomNum));
    		 logloss += Math.log10(normProb(knn.distributionForInstance(datatest.instance(randomNum))[(int)datatest.instance(randomNum).classValue()]))/-length;
    		 
    		 datatest.instance(randomNum).setClassValue(score);
    		 data.add(datatest.instance(randomNum));   		
    		 datatest.delete(randomNum);
    		 if(datatest.numInstances() % (length/100) == 0)System.out.print(".");	 
    	}
    	System.out.println("!\n");
    	return logloss;
    }
    
    
public static void semisupervisedknnSUBMISSION(Instances data, Instances datatest) throws Exception{
    	
    	Random rand = new Random();
    	int length = datatest.numInstances();
    	CSVWriter writer = new CSVWriter("./MachineLearning/result_knn.csv"); 
    	
    	while(datatest.numInstances() != 0)
    	{	//setup
    		 IBk knn = new IBk();
    		 knn.buildClassifier(data);
    		 int randomNum = rand.nextInt(datatest.numInstances());
    		 double score = knn.classifyInstance(datatest.instance(randomNum)); 
    		 double[] vv = knn.distributionForInstance(datatest.instance(randomNum));
    		 writer.addInstance((int)datatest.instance(randomNum).value(0), vv);
    		 datatest.instance(randomNum).setClassValue(score);
    		 data.add(datatest.instance(randomNum));   		
    		 datatest.delete(randomNum);
    		 if(datatest.numInstances() % (length/100) == 0)System.out.print(".");	 
    	}
    	writer.closeFile();
    	System.out.println("!\n");
    }

    public static double normProb(double probability){
    	double eps = Math.pow(10, -15);
    	return Math.max(Math.min(1-eps,  probability), eps);
    }
    
    
    public static void saveInstances(Instances dataSet, String fileName) throws IOException
    	{
    		ArffSaver Asaver = new ArffSaver();
    		Asaver.setInstances(dataSet);
    		Asaver.setFile(new File(fileName));
    		Asaver.writeBatch();
    	}
    
    public static Instances getResultforTestdata(Instances testOrig, Instances trainAfterClassification)
    	{
    		Instances InstancesLabeled = new Instances(testOrig, testOrig.numInstances());
    		trainAfterClassification.sort(0);
    
    		for (int i = 0; i<testOrig.numInstances(); i++)
    		{
    			int idOrig = (int)testOrig.instance(i).value(0);
    			System.out.println(idOrig);
    			Instance temp = trainAfterClassification.instance(idOrig-1);
    			int idTrain = (int)temp.value(0);
    			System.out.println(idTrain);
    
    			if (idOrig == idTrain)
    			{
    				InstancesLabeled.add(temp);
    			}    		
    		}
    
    		assert testOrig.numInstances() == InstancesLabeled.numInstances();
    
    		return InstancesLabeled;
    	}
    
    public static void main(String[] args) throws Exception{
    	
    	boolean knnBool = false;
    	boolean svmBool = false;
    	boolean semBool = true;
    	
    	boolean submissionOutput = true;
    	File res = new File("results.txt");
    	FileWriter result_writer = new FileWriter(res, true);
    	Date now = new Date();
    	result_writer.write("--- " + now.toString() + " ---\n");
    	
    	// read data from file
    	DataSource source = new DataSource("./MachineLearning/train.csv");
		DataSource sourcetest = new DataSource("./MachineLearning/test.csv");
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes()-1);
    	Instances datatest = sourcetest.getDataSet();
    	
    	System.out.println("Loaded datasets!");
    	
    	//add empty class label to test data
    	
        Add filter = new Add();
        filter.setAttributeIndex("last");
        filter.setNominalLabels("Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9");
        filter.setAttributeName("target");
        filter.setInputFormat(datatest);
        datatest = filter.useFilter(datatest, filter);
    	for ( int i = 0; i < datatest.numInstances(); i++){
    		//---!!PROBABLY NEEDS TO BE SET TO ?!!---
    		datatest.instance(i).setValue(datatest.numAttributes() - 1,"Class_1");
    	}
        datatest.setClassIndex(datatest.numAttributes()-1);
    	
        //attribute selection
        PrincipalComponents selector = new PrincipalComponents();
        selector.setInputFormat(data);
        data = selector.useFilter(data, selector);
        datatest = selector.useFilter(datatest, selector);
        
        System.out.println("Attributes Selected!");
        
        Random rand = new Random(1234);
    	Instances knnTrain = data.trainCV(2,0,rand);
    	Instances knnTest = data.trainCV(2, 1, rand);
    	Instances svmTrain = data.trainCV(2,0,rand);
    	Instances svmTest = data.trainCV(2, 1, rand);
        
   if(knnBool){
	    System.out.println("NN Classifier!");
        IBk knnClassifier = new IBk();
        									//Evaluation knnev = new Evaluation(data);
        									//knnev.crossValidateModel(knnClassifier, data, 2, new Random(1));
        									//System.out.println(knnev.toSummaryString());
        //calc logloss
        double logloss = 0;
        knnClassifier.buildClassifier(knnTrain);
        for(int i = 0; i < knnTest.numInstances();i++)
        {
            logloss += Math.log10(normProb(knnClassifier.distributionForInstance(knnTest.instance(i))[(int)knnTest.instance(i).classValue()]))/-knnTest.numInstances();
           
            if(i%(knnTest.numInstances()/100) == 0)System.out.print(".");
        }
    	System.out.println("!");
    	System.out.println("1/2 Knn Logloss is: " + logloss);
    	result_writer.write("1/2 Knn Logloss is: " + logloss + "\n");
    	
    	 logloss = 0;
         knnClassifier.buildClassifier(knnTest);
         for(int i = 0; i < knnTrain.numInstances();i++)
         {      		
             logloss += Math.log10(normProb(knnClassifier.distributionForInstance(knnTrain.instance(i))[(int)knnTrain.instance(i).classValue()]))/-knnTrain.numInstances();
            
             if(i%(knnTest.numInstances()/100) == 0)System.out.print(".");
         }
     	System.out.println("!");
     	System.out.println("2/2 Knn Logloss is: " + logloss);
     	result_writer.write("2/2 Knn Logloss is: " + logloss + "\n");
     	
        if(submissionOutput){
        	CSVWriter writer_knn = new CSVWriter("./MachineLearning/result_knn.csv"); 
        	knnClassifier.buildClassifier(data);
        	for(int i = 0; i < datatest.numInstances();i++)
            {
                double[] vv= knnClassifier.distributionForInstance(datatest.instance(i));
                writer_knn.addInstance(i, vv);
                if(i%(datatest.numInstances()/100) == 0)System.out.print(".");
                //datatest.instance(i).setClassValue(score);
            }
        	System.out.println("!");
        	writer_knn.closeFile();
        }
   }
   if(svmBool){    
	   System.out.println("SVM Classifier");
        LibSVM svmClassifier = new LibSVM();   
												//        Evaluation svmev = new Evaluation(data);
												//        svmev.crossValidateModel(svmClassifier, data, 2, new Random(1));
												//        System.out.println(svmev.toSummaryString());
        //calc logloss
        double logloss = 0;
        svmClassifier.buildClassifier(knnTrain);
        for(int i = 0; i < knnTest.numInstances();i++)
        {
            logloss += Math.log10(normProb(svmClassifier.distributionForInstance(knnTest.instance(i))[(int)knnTest.instance(i).classValue()]))/-knnTest.numInstances();
           
            if(i%(knnTest.numInstances()/100) == 0)System.out.print(".");
        }
    	System.out.println("!");
    	System.out.println("1/2 SVM Logloss is: " + logloss);
    	result_writer.write("1/2 SVM Logloss is: " + logloss + "\n");
    	
    	logloss = 0;
        svmClassifier.buildClassifier(knnTest);
        for(int i = 0; i < knnTrain.numInstances();i++)
        {
            logloss += Math.log10(normProb(svmClassifier.distributionForInstance(knnTrain.instance(i))[(int)knnTrain.instance(i).classValue()]))/-knnTrain.numInstances();
           
            if(i%(knnTrain.numInstances()/100) == 0)System.out.print(".");
        }
    	System.out.println("!");
    	System.out.println("2/2 SVM Logloss is: " + logloss);
    	result_writer.write("2/2 SVM Logloss is: " + logloss + "\n");
        
        if(submissionOutput){
        	CSVWriter writer_svm = new CSVWriter("./MachineLearning/result_svm.csv"); 
        	svmClassifier.buildClassifier(data);
        	for(int i = 0; i < datatest.numInstances();i++)
            {
                double[] vv= svmClassifier.distributionForInstance(datatest.instance(i));
                writer_svm.addInstance(i, vv);
                if(i%(datatest.numInstances()/100) == 0)System.out.print(".");
            }
        	System.out.println("!");
        	writer_svm.closeFile();
        }
   }
   if(semBool){
	   	System.out.println("Semi Classifier");
        //semisupervised classifier
   		Instances semiTrainTE = data.trainCV(2,0,rand);
   		Instances semiTrain = semiTrainTE.trainCV(2,0,rand);
   		Instances semiTest = semiTrainTE.trainCV(2, 1, rand);
    	
   		Instances semiTrain2 = semiTrainTE.trainCV(2,0,rand);;
   		Instances SemiTest2 = semiTrainTE.trainCV(2,1,rand);;
   		
   		System.out.println("TRAINSIZE: " + semiTrain.numInstances()+" TESTSIZE: "  + semiTest.numInstances());
   		double fold1 = semisupervisedknn(semiTrain, semiTest);
    	System.out.println("1/2 Sem Logloss is: " + fold1 + "\n");
    	result_writer.write("1/2 Sem Logloss is: " + fold1 + "\n");
    	System.out.println("TRAINSIZE2: " + semiTrain2.numInstances()+" TESTSIZE2: "  + SemiTest2.numInstances());
    	
    	double fold2 = semisupervisedknn(SemiTest2, semiTrain2);
    	System.out.println("2/2 Sem Logloss is: " + fold2 + "\n");
    	result_writer.write("2/2 Sem Logloss is: " + fold2 + "\n");
    	if(submissionOutput){
    		
    	semisupervisedknnSUBMISSION(data, datatest);
    		
    	}
   } 	
   		Date end = new Date();
   		result_writer.write("-+-"+ end.toString() + "-+-");
   		result_writer.close();
       
    }
}
