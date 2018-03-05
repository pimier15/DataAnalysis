using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using XGBoost.lib;
using XGBoost;
using static XGBoost.XGBRegressor;

namespace AnalysisCs
{
	class Program
	{
		static void Main( string[] args )
		{
			float[][] train = new float[][]
				{
					new float[] { 1,2},
					new float[] { 2,2},
					new float[] { 3,2},
					new float[] { 10,20},
					new float[] { 12,44}
				};
			float[][] train2 = new float[][]
				{
					new float[] { 6,3},
					new float[] { 2,9},
					new float[] { 8,10},
					new float[] { 19,29},
					new float[] { 29,24}
				};

			float[] lable = new float[] { 0 ,0 , 0 ,1 , 1};


			var reg = new XGBRegressor();
			reg.Fit( train, lable );

			var temp1 = reg.Predict(train);
			var temp12 = reg.Predict(train2);

			Console.WriteLine();





		}
	}

	public static class ext
	{

	}

}
