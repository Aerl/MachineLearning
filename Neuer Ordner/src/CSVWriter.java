import java.io.FileWriter;
import java.io.IOException;


public class CSVWriter {
	String FilePath;
	FileWriter writer;

	public CSVWriter(String path) 
	{
		FilePath = path;
		try 
		{
			writer = new FileWriter(FilePath); 
			writer.append("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9");
			writer.append('\n');
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}

	public void addInstance(int id, double[] distribution) 
	{
		try 
		{
			writer.append(String.valueOf(id+1));
			for (double probability : distribution)
			{
				writer.append(',');
				writer.append(String.valueOf(probability));
			}
			writer.append('\n');
		}
		catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void closeFile() 
	{
		try 
		{
			writer.close();
		}
		catch (IOException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
