	[?{c?Tx@[?{c?Tx@![?{c?Tx@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'[?{c?Tx@???+?z@10??{xx@Ic??V???r0*	333333Y@2_
(Iterator::Root::MapAndBatch::TensorSlice?@??ǘ??!?q?q?J@)?@??ǘ??1?q?q?J@:Preprocessing2R
Iterator::Root::MapAndBatch7?[ A??!v]?u]?@@)7?[ A??1v]?u]?@@:Preprocessing2E
Iterator::Root?c]?F??!9??8?CG@)F%u???1?0?0*@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI '?n???Q??"? ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???+?z@???+?z@!???+?z@      ??!       "	0??{xx@0??{xx@!0??{xx@*      ??!       2      ??!       :	c??V???c??V???!c??V???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q '?n???y??"? ?X@