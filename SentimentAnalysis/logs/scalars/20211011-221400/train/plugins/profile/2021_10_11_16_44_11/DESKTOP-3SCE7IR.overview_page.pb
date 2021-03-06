?	?rJߤ~@?rJߤ~@!?rJߤ~@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?rJߤ~@????@1?!???|@I??v?@r0*	fffff?N@2_
(Iterator::Root::MapAndBatch::TensorSlice????Mb??!N??o?J@)????Mb??1N??o?J@:Preprocessing2R
Iterator::Root::MapAndBatchˡE?????!?(?W??@@)ˡE?????1?(?W??@@:Preprocessing2E
Iterator::Root?X?? ??!???G@)/n????1?????,@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?ú??@Q?S?$@W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????@????@!????@      ??!       "	?!???|@?!???|@!?!???|@*      ??!       2      ??!       :	??v?@??v?@!??v?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?ú??@y?S?$@W@?"Y
;model/bert/encoder/layer_._10/output/dense/Tensordot/MatMulMatMul;?????!;?????0"X
:model/bert/encoder/layer_._3/output/dense/Tensordot/MatMulMatMul{??Cπ??!?[ct???0"P
3model/bert/encoder/layer_._4/attention/self/SoftmaxSoftmax{?'?"[??!̟{d???"X
:model/bert/encoder/layer_._1/output/dense/Tensordot/MatMulMatMul??ҹ????!qD?R
??0"Q
4model/bert/encoder/layer_._10/attention/self/SoftmaxSoftmaxN?j???!D?q????"X
:model/bert/encoder/layer_._9/output/dense/Tensordot/MatMulMatMul?~?с?!?(?b??0"^
@model/bert/encoder/layer_._5/intermediate/dense/Tensordot/MatMulMatMul??K?8??!`??s??0"X
:model/bert/encoder/layer_._2/output/dense/Tensordot/MatMulMatMul`0???!??}???0"K
0model/bert/encoder/layer_._11/attention/self/AddAddV2?V??A:?!:?s??"J
/model/bert/encoder/layer_._5/attention/self/AddAddV2?1????|?!*"?ӷ?Q      Y@Y????T@aq??5??W@qR?bVo?X@y'>?NV?"?

both?Your program is POTENTIALLY input-bound because 6.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?99.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 