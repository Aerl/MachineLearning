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

public class test {
 
    public static void main(String[] args) throws Exception{
    	// read data from file
    	DataSource source = new DataSource("C:/train.csv");
    	DataSource sourcetest = new DataSource("C:/test.csv");
    	Instances datatest = sourcetest.getDataSet();
    	System.out.println("loaded testdata");
    	Instances data = source.getDataSet();
    	System.out.println("loaded traindata");
    	
    	//set class attribute
    	data.setClassIndex(data.numAttributes()-1);
    	
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
        
        System.out.println("added -1 class label to testdata");
        
        //attribute selecetion
        AttributeSelection selector = new AttributeSelection();
        selector.setInputFormat(data);
        data = selector.useFilter(data, selector);
        datatest = selector.useFilter(datatest, selector);
        
        System.out.println("reduced dims");
        /*
    	IBk k = new IBk();

    	k.buildClassifier(data);
    	for( int i = 0; i < k.getOptions().length; i++)
    		System.out.println(k.getOptions()[i]);
    	for(int i = 0; i < datatest.numInstances();i++)
        {
    		
            double score = k.classifyInstance(datatest.instance(i));
            //double[] vv= k.distributionForInstance(datatest.instance(i));
            System.out.println(i + ": class: "+ score);
            datatest.instance(i).setClassValue(score);
            
        }
    	
        System.out.println("classified datatest");
        */

		 ArffSaver Asaver = new ArffSaver();
		 Asaver.setInstances(datatest);
		 Asaver.setFile(new File("./MachineLearning/testminusone.arff"));
		 //Asaver.setDestination(new File("./data/test.arff"));   // **not** necessary in 3.5.4 and later
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
    }
}
