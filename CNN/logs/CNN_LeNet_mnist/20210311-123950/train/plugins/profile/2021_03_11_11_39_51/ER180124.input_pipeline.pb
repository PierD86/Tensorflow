	r???gO@r???gO@!r???gO@	??V?????V???!??V???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$r???gO@?	h"lx??A?Zd;GO@YU???N@??*	33333?j@2U
Iterator::Model::ParallelMapV2?????̼?!?JV??UJ@)?????̼?1?JV??UJ@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Q?????!????b0@)???Q???1?\?,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipjM??St??!W???B@)2??%䃞?1?H?7?+@:Preprocessing2F
Iterator::Model[Ӽ???!??/*?KO@)?g??s???1~?fߥ?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???߾??!;)?Nʧ@)HP?sׂ?1????|:@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!????@){?G?zt?1????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;?O??nr?!NmjS?? @);?O??nr?1NmjS?? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?W[?????!%???F@)Ǻ???V?1P'??I???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??V???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?	h"lx???	h"lx??!?	h"lx??      ??!       "      ??!       *      ??!       2	?Zd;GO@?Zd;GO@!?Zd;GO@:      ??!       B      ??!       J	U???N@??U???N@??!U???N@??R      ??!       Z	U???N@??U???N@??!U???N@??JCPU_ONLYY??V???b 