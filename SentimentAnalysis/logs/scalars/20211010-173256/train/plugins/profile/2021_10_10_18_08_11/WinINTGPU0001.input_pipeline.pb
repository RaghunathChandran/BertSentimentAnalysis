	C??gWx@C??gWx@!C??gWx@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'C??gWx@????@1??̓+x@ID?K?K???r0*	??????]@2_
(Iterator::Root::MapAndBatch::TensorSlice?????ױ?!<?h?0M@)?????ױ?1<?h?0M@:Preprocessing2R
Iterator::Root::MapAndBatch?Q?????!?h?0P=@)?Q?????1?h?0P=@:Preprocessing2E
Iterator::Root5?8EGr??!?^?#??D@)%u???1ϩ????(@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI??}?s??Q??	N0?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????@????@!????@      ??!       "	??̓+x@??̓+x@!??̓+x@*      ??!       2      ??!       :	D?K?K???D?K?K???!D?K?K???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??}?s??y??	N0?X@