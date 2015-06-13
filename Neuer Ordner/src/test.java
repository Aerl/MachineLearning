import java.io.File;

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
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.LibSVM;

import java.io.FileWriter;
import java.util.Random;

public class test {


	public static Instances semisupervisedknn(Instances data, Instances datatest) throws Exception
	{

		Instances datatestOrig = new Instances(datatest);
		
		Random rand = new Random();

		while(datatest.numInstances() != 0)
		{
			IBk knn = new IBk();
			knn.buildClassifier(data);
			int randomNum = rand.nextInt((datatest.numInstances() - 0) + 1);
			double score = knn.classifyInstance(datatest.instance(randomNum));
			datatest.instance(randomNum).setClassValue(score);
			data.add(datatest.instance(randomNum));   		
			datatest.delete(randomNum);
			if(datatest.numInstances() % 1000 == 0)System.out.println(datatest.numInstances() + ":" + randomNum + ":" + score);
		}
		
		return getResultforTestdata(datatestOrig, data);

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

	public static void main(String[] args) throws Exception
	{

		boolean knnBool = true;
		boolean svmBool = false;
		boolean semBool = false;

		double eps = Math.pow(10, -15);

		boolean submissionOutput = false;

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
		for ( int i = 0; i < datatest.numInstances(); i++)
		{
			datatest.instance(i).setValue(datatest.numAttributes() - 1,"Class_1");
		}
		datatest.setClassIndex(datatest.numAttributes()-1);

		System.out.println("added class label to testdata");

		//attribute selection
		AttributeSelection selector = new AttributeSelection();
		selector.setInputFormat(data);
		data = selector.useFilter(data, selector);
		datatest = selector.useFilter(datatest, selector);

		System.out.println("Attributes Selected!");


		if(knnBool){
			System.out.println("NN Classifier!");
			//knn classifier
			IBk knnClassifier = new IBk();
			Evaluation knnev = new Evaluation(data);
			knnev.crossValidateModel(knnClassifier, data, 2, new Random(1));
			System.out.println(knnev.toSummaryString());

			//TODO: calc logloss

			if(submissionOutput){
				CSVWriter writer_knn = new CSVWriter("./MachineLearning/result_knn.csv"); 
				knnClassifier.buildClassifier(data);
				for(int i = 0; i < datatest.numInstances();i++)
				{
					double[] vv= knnClassifier.distributionForInstance(datatest.instance(i));
					writer_knn.addInstance(i, vv);
					if(i%(datatest.numInstances()/100) == 0)System.out.print(i);
					//datatest.instance(i).setClassValue(score);
				}
				System.out.println("!");
				writer_knn.closeFile();
			}
		}
		if(svmBool){    
			System.out.println("SVM Classifier");
			//svm classifier
			LibSVM svmClassifier = new LibSVM();

			Evaluation svmev = new Evaluation(data);
			svmev.crossValidateModel(svmClassifier, data, 2, new Random(1));
			System.out.println(svmev.toSummaryString());

			if(submissionOutput){
				CSVWriter writer_svm = new CSVWriter("./MachineLearning/result_svm.csv"); 
				svmClassifier.buildClassifier(data);
				for(int i = 0; i < datatest.numInstances();i++)
				{
					double[] vv= svmClassifier.distributionForInstance(datatest.instance(i));
					writer_svm.addInstance(i, vv);
					if(i%(datatest.numInstances()/100) == 0)System.out.print(i);
					//datatest.instance(i).setClassValue(score);
				}
				System.out.println("!");
				writer_svm.closeFile();
			}
		}
		if(semBool){
			System.out.println("Semi Classifier");
			//semisupervised classifier
			Random rand = new Random(1234);
			Instances semiTrain = data.trainCV(2,1,rand);
			Instances semiTest = data.trainCV(2, 1, rand);

			semisupervisedknn(semiTrain, semiTest);
		} 	
	}

}
