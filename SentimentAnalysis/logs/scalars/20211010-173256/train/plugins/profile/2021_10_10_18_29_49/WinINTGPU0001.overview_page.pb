?	[?{c?Tx@[?{c?Tx@![?{c?Tx@      ??!       "h
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
	???+?z@???+?z@!???+?z@      ??!       "	0??{xx@0??{xx@!0??{xx@*      ??!       2      ??!       :	c??V???c??V???!c??V???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q '?n???y??"? ?X@?"X
:model/bert/encoder/layer_._6/output/dense/Tensordot/MatMulMatMulHb???&??!Hb???&??0"X
:model/bert/encoder/layer_._5/output/dense/Tensordot/MatMulMatMulu--N1!??!?Ǌ??#??0"X
:model/bert/encoder/layer_._8/output/dense/Tensordot/MatMulMatMulB???&!??! J??IZ??0"X
:model/bert/encoder/layer_._0/output/dense/Tensordot/MatMulMatMul?????!?!?!??0"X
:model/bert/encoder/layer_._3/output/dense/Tensordot/MatMulMatMul5?C=0??!?9c???0"X
:model/bert/encoder/layer_._9/output/dense/Tensordot/MatMulMatMulUC?????!?L-"?W??0"X
:model/bert/encoder/layer_._7/output/dense/Tensordot/MatMulMatMul&6?{??!??ù??0"Y
;model/bert/encoder/layer_._11/output/dense/Tensordot/MatMulMatMulJ??d8??!?$?????0"X
:model/bert/encoder/layer_._4/output/dense/Tensordot/MatMulMatMul?Ӱ???!e??`)}??0"X
:model/bert/encoder/layer_._1/output/dense/Tensordot/MatMulMatMulp?^Τ ??!?7K?????0Q      Y@Y????T@aq??5??W@qV???Y?X@"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?99.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 