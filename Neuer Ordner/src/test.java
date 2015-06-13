import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

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

public class test {

	public static void main(String[] args) throws Exception{
		// read data from file
		DataSource source = new DataSource("./MachineLearning/train.csv");
		Instances data = source.getDataSet();
		System.out.println("loaded traindata");

		DataSource sourcetest = new DataSource("./MachineLearning/testminusone.csv");
		Instances datatest = sourcetest.getDataSet();
		System.out.println("loaded testdata");

		Instances help = getResultforTestdata(data, data);

		//set class attribute
		data.setClassIndex(data.numAttributes()-1);
		datatest.setClassIndex(data.numAttributes()-1);

		/*

    	// add dummy class to unlabeled
        Add filter = new Add();
        filter.setAttributeIndex("last");
        filter.setNominalLabels("Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9,");
        filter.setAttributeName("target");
        filter.setInputFormat(datatest);
        datatest = filter.useFilter(datatest, filter);
    	for ( int i = 0; i < datatest.numInstances(); i++){

    		datatest.instance(i).setValue(datatest.numAttributes() - 1,"Class_1");
    	}
        datatest.setClassIndex(datatest.numAttributes()-1);

        System.out.println("added class label to testdata");

         ArffSaver Asaver = new ArffSaver();
		 Asaver.setInstances(datatest);
		 Asaver.setFile(new File("./MachineLearning/testminusone.arff"));
		 Asaver.writeBatch();

		CSVSaver Csaver = new CSVSaver();
		Csaver.setInstances(datatest);

		String[] newOp = new String[4];

		newOp[0] = "-i";
		newOp[1] = "./MachineLearning/testminusone.arff";
		newOp[2] = "-o";
		newOp[3] = "./MachineLearning/testminusone.csv";

		Csaver.setOptions(newOp);
		//saver.setFile(new File("./MachineLearning/test.csv"));
		//saver.setDestination(new File("./data/test.arff"));   // **not** necessary in 3.5.4 and later
		Csaver.writeBatch();
		 */


		//attribute selection
		AttributeSelection selector = new AttributeSelection();
		selector.setInputFormat(data);
		data = selector.useFilter(data, selector);
		datatest = selector.useFilter(datatest, selector);

		System.out.println("reduced dims");

		/*
        LibSVM k = new LibSVM();

    	k.buildClassifier(data);

    	weka.core.SerializationHelper.write("./MachineLearning/classifier.model", k);
		 */

		ObjectInputStream ois = new ObjectInputStream(new FileInputStream("./MachineLearning/classifier.model"));
		Classifier k = (Classifier) ois.readObject();
		ois.close();

		CSVWriter writer = new CSVWriter("./MachineLearning/result.csv"); 

		for(int i = 0; i < datatest.numInstances();i++)
		{
			//double[] score = k.distributionForInstance(datatest.instance(i));
			double score = k.classifyInstance(datatest.instance(i));
			double[] vv= k.distributionForInstance(datatest.instance(i));
			writer.addInstance(i, vv);
			if (i%1000==0)System.out.println(i);
			//datatest.instance(i).setClassValue(score);

		}

		writer.closeFile();

		System.out.println("classified datatest");

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
}
