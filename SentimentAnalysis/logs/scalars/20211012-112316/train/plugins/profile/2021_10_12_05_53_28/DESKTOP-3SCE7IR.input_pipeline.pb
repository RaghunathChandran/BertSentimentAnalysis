	??
~?,{@??
~?,{@!??
~?,{@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??
~?,{@?mr??-@1???_?z@I?YO?? @r0*	??? ?rL@2_
(Iterator::Root::MapAndBatch::TensorSlice??cϞ?!???1?pJ@)??cϞ?1???1?pJ@:Preprocessing2R
Iterator::Root::MapAndBatchٰ??(???!?
*^A=@@)ٰ??(???1?
*^A=@@:Preprocessing2E
Iterator::Root?m?s??!t^x??G@)n??t???1	N9?)G-@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?bY?	@Q??7?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?mr??-@?mr??-@!?mr??-@      ??!       "	???_?z@???_?z@!???_?z@*      ??!       2      ??!       :	?YO?? @?YO?? @!?YO?? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?bY?	@y??7?X@