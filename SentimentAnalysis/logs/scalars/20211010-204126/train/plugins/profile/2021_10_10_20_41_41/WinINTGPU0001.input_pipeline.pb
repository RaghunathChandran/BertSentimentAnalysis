	0???]x@0???]x@!0???]x@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'0???]x@?:????@1?Z|
 x@I9?)9'v??r0*	?????[@2_
(Iterator::Root::MapAndBatch::TensorSlice???9#J??!3????H@)???9#J??13????H@:Preprocessing2R
Iterator::Root::MapAndBatchsh??|???!Мw^$C@)sh??|???1Мw^$C@:Preprocessing2E
Iterator::RootK?46??!?ip?EjI@)_?Qڋ?1?3???)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?]=?7??Q??S ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?:????@?:????@!?:????@      ??!       "	?Z|
 x@?Z|
 x@!?Z|
 x@*      ??!       2      ??!       :	9?)9'v??9?)9'v??!9?)9'v??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?]=?7??y??S ?X@