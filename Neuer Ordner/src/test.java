import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.KStar;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Add;

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
