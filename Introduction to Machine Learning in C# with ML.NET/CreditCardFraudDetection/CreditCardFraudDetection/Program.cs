using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using CreditCardFraudDetection.DataModels;

namespace CreditCardFraudDetection
{
    class Program
    {

        static void Main(string[] args)
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            string MODEL_FILEPATH = Path.Combine(assetsPath, "model.zip");
            string DATA_FILEPATH = Path.Combine(assetsPath, "creditcard.csv");

            MLContext mlContext = new MLContext();

            // Creating, training and saving model
            ModelBuilder.CreateModel(DATA_FILEPATH, MODEL_FILEPATH);

            // Loading model
            ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(MODEL_FILEPATH), out DataViewSchema inputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Get single data-point
            ModelInput sampleData = CreateSingleDataSample(mlContext, DATA_FILEPATH, 0);

            ModelOutput predictionResult = predEngine.Predict(sampleData);

            Console.WriteLine($"Actual value: {sampleData.Class} | Predicted value: {predictionResult.Prediction}");
            Console.ReadKey();
        }

        private static ModelInput CreateSingleDataSample(MLContext mlContext, string dataFilePath, int index)
        {       
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: dataFilePath,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            ModelInput sampleForPrediction = mlContext.Data.CreateEnumerable<ModelInput>(dataView, false).ElementAt(index);
                                                                        
            return sampleForPrediction;
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
