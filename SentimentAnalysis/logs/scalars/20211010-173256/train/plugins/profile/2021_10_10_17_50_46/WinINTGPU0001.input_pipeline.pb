	????s]x@????s]x@!????s]x@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'????s]x@???^D?@1?w???!x@I???????r0*	?????YY@2_
(Iterator::Root::MapAndBatch::TensorSlice??e?c]??!?7<KQK@)??e?c]??1?7<KQK@:Preprocessing2R
Iterator::Root::MapAndBatch????Mb??!۽=???@)????Mb??1۽=???@:Preprocessing2E
Iterator::RootZd;?O???!u??ô?F@)y?&1???1 ?u??+@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?1?KT???Q?MhW??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???^D?@???^D?@!???^D?@      ??!       "	?w???!x@?w???!x@!?w???!x@*      ??!       2      ??!       :	??????????????!???????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?1?KT???y?MhW??X@