	??K7?yJ@??K7?yJ@!??K7?yJ@	JC?B!???JC?B!???!JC?B!???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??K7?yJ@ݵ?|г??ApΈ???I@Y!?rh????*	??????u@2F
Iterator::ModelQk?w????!
ϭ?V@)Έ?????1[I??fgU@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!?ґ=Q@)?&S???1??????@:Preprocessing2U
Iterator::Model::ParallelMapV2HP?sׂ?!??y?{,@)HP?sׂ?1??y?{,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea??+e??!?a?jމ@)? ?	??1q?w??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU???N@s?!??|?Q???)U???N@s?1??|?Q???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ZӼ???!?7???y'@);?O??nr?1?Ow?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!o?!?db?}??)ŏ1w-!o?1?db?}??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???QI??!?Ȟ??t@)ŏ1w-!_?1?db?}??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9JC?B!???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ݵ?|г??ݵ?|г??!ݵ?|г??      ??!       "      ??!       *      ??!       2	pΈ???I@pΈ???I@!pΈ???I@:      ??!       B      ??!       J	!?rh????!?rh????!!?rh????R      ??!       Z	!?rh????!?rh????!!?rh????JCPU_ONLYYJC?B!???b 