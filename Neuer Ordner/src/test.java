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

public class test {
 
    public static void main(String[] args) throws Exception{
    	// read data from file
    	DataSource source = new DataSource("C:/train.csv");
    	DataSource sourcetest = new DataSource("C:/test.csv");
    	Instances datatest = sourcetest.getDataSet();
    	Instances data = source.getDataSet();
    	//set class attribute
    	data.setClassIndex(data.numAttributes()-1);
    	
    	// 2. numeric attribute
        Add filter = new Add();
        filter.setAttributeIndex("last");
        filter.setAttributeName("target1");
        filter.setInputFormat(datatest);
        datatest = filter.useFilter(datatest, filter);
    	for ( int i = 0; i < datatest.numInstances(); i++){
    		datatest.instance(i).setValue(datatest.numAttributes() - 1, -1);
    	}
        datatest.setClassIndex(datatest.numAttributes()-1);
        

		 ArffSaver Asaver = new ArffSaver();
		 Asaver.setInstances(datatest);
		 Asaver.setFile(new File("./MachineLearning/test.arff"));
		 //Asaver.setDestination(new File("./data/test.arff"));   // **not** necessary in 3.5.4 and later
		 Asaver.writeBatch();
        
		CSVSaver Csaver = new CSVSaver();
		Csaver.setInstances(datatest);
		
		String[] newOp = new String[4];
		
		newOp[0] = "-i";
		newOp[1] = "./MachineLearning/test.arff";
		newOp[2] = "-o";
		newOp[3] = "./MachineLearning/test.csv";
		
		Csaver.setOptions(newOp);
		//saver.setFile(new File("./MachineLearning/test.csv"));
		//saver.setDestination(new File("./data/test.arff"));   // **not** necessary in 3.5.4 and later
		Csaver.writeBatch();
		
		
        /*KStar k = new KStar();
    	k.buildClassifier(data);
    	System.out.println("loaded");
    	for(int i = 0; i < datatest.numInstances();i++)
        {
    		
            double score = k.classifyInstance(datatest.instance(i));
            double[] vv= k.distributionForInstance(datatest.instance(i));
            System.out.println("class");
            
        }*/
    	
        System.out.println("dunno");
    }
}
