	0e???Mx@0e???Mx@!0e???Mx@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'0e???Mx@m??3?/@1?u??x@I??ߠ?z??r0*	43333X@2_
(Iterator::Root::MapAndBatch::TensorSlice??C?l???!??qLL@)??C?l???1??qLL@:Preprocessing2R
Iterator::Root::MapAndBatch6?;Nё??!?/Ċ??<@)6?;Nё??1?/Ċ??<@:Preprocessing2E
Iterator::Root??A?f??!a?|???E@)??Pk?w??1?ek$=?,@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?[R?mI??Q??BHڶX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	m??3?/@m??3?/@!m??3?/@      ??!       "	?u??x@?u??x@!?u??x@*      ??!       2      ??!       :	??ߠ?z????ߠ?z??!??ߠ?z??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?[R?mI??y??BHڶX@