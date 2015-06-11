import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.KStar;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Add;
import weka.core.converters.CSVSaver;
import weka.core.converters.ArffSaver;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

public class test {
 
    public static void main(String[] args) throws Exception
    {
		// read data from file
		DataSource source = new DataSource("./MachineLearning/train.csv");
		DataSource sourcetest = new DataSource("./MachineLearning/test.csv");
		Instances datatest = sourcetest.getDataSet();
		System.out.println("loaded testdata");
		Instances data = source.getDataSet();
		System.out.println("loaded traindata");
		
		//set class attribute
		data.setClassIndex(data.numAttributes()-1);
		// add dummy class to unlabeled
		Add filter = new Add();
		filter.setAttributeIndex("last");
		filter.setAttributeName("target");
		//filter.setAttributeType((FastVector) null);
		
		String[] old = filter.getOptions();
		String[] op = new String[6];
		op[0] = old[0];
		op[1] = old[1];
		op[2] = old[2];
		op[3] = old[3];
		op[4] = "-T";
		op[5] = "STR";
		filter.setOptions(op);
		
		filter.setInputFormat(datatest);
		datatest = filter.useFilter(datatest, filter);
		for ( int i = 0; i < datatest.numInstances(); i++)
		{
			datatest.instance(i).setValue(datatest.numAttributes() - 1, "Class_0");
			//System.out.println(datatest.instance(i).attribute(datatest.numAttributes() - 1));
		}
		datatest.setClassIndex(datatest.numAttributes()-1);
		
		  // Safe to CSV
		ArffSaver Asaver = new ArffSaver();
		Asaver.setInstances(datatest);
		Asaver.setFile(new File("./MachineLearning/save.arff"));
		Asaver.writeBatch();
		
		CSVSaver Csaver = new CSVSaver();
		Csaver.setInstances(datatest);
		
		String[] newOp = new String[4];
		
		newOp[0] = "-i";
		newOp[1] = "./MachineLearning/save.arff";
		newOp[2] = "-o";
		newOp[3] = "./MachineLearning/save.csv";
		
		Csaver.setOptions(newOp);
		Csaver.writeBatch();		
     
		
		
			   
	    System.out.println("added -1 class label to testdata");
	
	    /*		
		//attribute selecetion
		AttributeSelection selector = new AttributeSelection();
		selector.setInputFormat(data);
		data = selector.useFilter(data, selector);
		datatest = selector.useFilter(datatest, selector);
		
		System.out.println("reduced dims");
		*/
		
		J48 k = new J48();
		
		k.buildClassifier(data);

		// evaluate classifier and print some statistics
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(k, datatest);

		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		
/*		for(int i = 0; i < datatest.numInstances();i++)
		{
			
		    double score = k.classifyInstance(datatest.instance(i));
		    double[] vv= k.distributionForInstance(datatest.instance(i));
		    System.out.println("class");
		    
		}*/
		
		System.out.println("dunno");
	}
    
    
}
