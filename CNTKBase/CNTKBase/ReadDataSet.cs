using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using Emgu.CV;

namespace CNTKBase
{
	class ReadDataSet
	{
	}

	public class Readmnist
	{
		string pathImg = @"F:\00_github\DataAnalysis\dataset\mnist\testdata\t10k-images.idx3-ubyte";
		string pathLabel = @"F:\00_github\DataAnalysis\dataset\mnist\testlabel\t10k-labels.idx1-ubyte";


		public Dictionary<string, byte[]> ReadRawmnist(string imgPath , string lbaelPath)
		{
			var testImgs = File.ReadAllBytes(pathImg).Skip(16).ToArray();
			var testLabels = File.ReadAllBytes(pathLabel).Skip(8).ToArray();
			Console.WriteLine();
		}

	}

}
