?	?]M??Wx@?]M??Wx@!?]M??Wx@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?]M??Wx@?\?&?[@1?.Ȗ x@I?W?????r0*	?????LZ@2_
(Iterator::Root::MapAndBatch::TensorSliceı.n???!t?&h??I@)ı.n???1t?&h??I@:Preprocessing2R
Iterator::Root::MapAndBatch'?Wʢ?!??4fqA@)'?Wʢ?1??4fqA@:Preprocessing2E
Iterator::RootB`??"۩?!?ٗ| H@)lxz?,C??1?$?Y<*@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI? 3mL??Q???%g?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?\?&?[@?\?&?[@!?\?&?[@      ??!       "	?.Ȗ x@?.Ȗ x@!?.Ȗ x@*      ??!       2      ??!       :	?W??????W?????!?W?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q? 3mL??y???%g?X@?"X
:model/bert/encoder/layer_._0/output/dense/Tensordot/MatMulMatMul??t???!??t???0"X
:model/bert/encoder/layer_._2/output/dense/Tensordot/MatMulMatMulD??????!t??R???0"X
:model/bert/encoder/layer_._6/output/dense/Tensordot/MatMulMatMul?{?????!?xW K??0"X
:model/bert/encoder/layer_._1/output/dense/Tensordot/MatMulMatMul?K?2???!???(@??0"X
:model/bert/encoder/layer_._3/output/dense/Tensordot/MatMulMatMul???????!W?j????0"Y
;model/bert/encoder/layer_._10/output/dense/Tensordot/MatMulMatMulA?Q???!e??J??0"X
:model/bert/encoder/layer_._7/output/dense/Tensordot/MatMulMatMul)?#?H??!*??????0"X
:model/bert/encoder/layer_._5/output/dense/Tensordot/MatMulMatMulI'u??
??!;%I???0"X
:model/bert/encoder/layer_._9/output/dense/Tensordot/MatMulMatMul??hС
??!?P2?-n??0"X
:model/bert/encoder/layer_._4/output/dense/Tensordot/MatMulMatMul??V???!?N?x???0Q      Y@Y????T@aq??5??W@qEػ??X@"?

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
Refer to the TF2 Profiler FAQb?99.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 