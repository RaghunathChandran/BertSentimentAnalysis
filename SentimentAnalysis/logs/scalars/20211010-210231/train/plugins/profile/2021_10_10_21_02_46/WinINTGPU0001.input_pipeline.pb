	??'G?,x@??'G?,x@!??'G?,x@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??'G?,x@???+?:@1Z??Uy?w@I??t?i%??r0*	53333?\@2_
(Iterator::Root::MapAndBatch::TensorSlice1?Zd??!ZɫG@gG@)1?Zd??1ZɫG@gG@:Preprocessing2R
Iterator::Root::MapAndBatch?c]?F??!??=?$?D@)?c]?F??1??=?$?D@:Preprocessing2E
Iterator::Rootŏ1w-!??!?6T???J@)??Pk?w??1|Z?kR(@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@???????Qo?p??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???+?:@???+?:@!???+?:@      ??!       "	Z??Uy?w@Z??Uy?w@!Z??Uy?w@*      ??!       2      ??!       :	??t?i%????t?i%??!??t?i%??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@???????yo?p??X@