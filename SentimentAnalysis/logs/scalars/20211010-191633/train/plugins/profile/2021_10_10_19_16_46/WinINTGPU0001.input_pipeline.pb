	????Vx@????Vx@!????Vx@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'????Vx@??]??@1????x@I??!?????r0*	33333cb@2_
(Iterator::Root::MapAndBatch::TensorSlice??ܵ??!?m???K@)??ܵ??1?m???K@:Preprocessing2R
Iterator::Root::MapAndBatch	?c???!?W7??A@)	?c???1?W7??A@:Preprocessing2E
Iterator::Rootm???{???!0?Hao?F@)_?Qڋ?1??D??}"@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@b~??f??Qw??e?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??]??@??]??@!??]??@      ??!       "	????x@????x@!????x@*      ??!       2      ??!       :	??!???????!?????!??!?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@b~??f??yw??e?X@