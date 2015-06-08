import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.KStar;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class test {
 
    public static void main(String[] args) throws Exception{
    	// read data from file
    	DataSource source = new DataSource("C:/Users/Bjrn/Desktop/SS2015/MachineLearning/train.csv");
    	DataSource sourcetest = new DataSource("C:/Users/Bjrn/Desktop/SS2015/MachineLearning/test.csv");
    	Instances datatest = sourcetest.getDataSet();
    	Instances data = source.getDataSet();
    	//set class attribute
    	data.setClassIndex(data.numAttributes()-1);
    	
    	KStar k = new KStar();
    	k.buildClassifier(data);
    	System.out.println("loaded");
    	for(int i = 0; i < datatest.numInstances();i++)
        {

            double score = k.classifyInstance(datatest.instance(i));
            double[] vv= k.distributionForInstance(datatest.instance(i));
            System.out.println("class");
            
        }
    	
        System.out.println("dunno");
    }
}
