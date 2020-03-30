import pyspark

sc = pyspark.SparkContext()
rdd = sc.parallelize(['Hello,', 'world!', 'dog', 'elephant', 'panther'])
res = rdd.collect()
print("res", res)
rdd.coalesce(1).saveAsTextFile(SPARK_BUCKET_OUTPUT_PATH)
