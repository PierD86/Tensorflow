	?lV}??K@?lV}??K@!?lV}??K@	51z??ѱ?51z??ѱ?!51z??ѱ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?lV}??K@??ʡE??A?!?uq?K@Y?j+??ݣ?*	gffff?M@2F
Iterator::Model?z6?>??!4Lu???B@)X?5?;N??1d!Y?B<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ׁsF??!sn?? ?@@)??ܵ?|??1?N??N?:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??ZӼ???!n?? ?1@)S?!?uq{?1?Gqth&@:Preprocessing2U
Iterator::Model::ParallelMapV2?????w?!????e#@)?????w?1????e#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??\m????!ͳ?#O@)n??t?1?yVQc @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???_vOn?!?8:4ɿ@)???_vOn?1?8:4ɿ@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?l?!?e?Oi@)y?&1?l?1?e?Oi@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa??+e??!???w\?4@)/n??b?1???rn@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no941z??ѱ?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ʡE????ʡE??!??ʡE??      ??!       "      ??!       *      ??!       2	?!?uq?K@?!?uq?K@!?!?uq?K@:      ??!       B      ??!       J	?j+??ݣ??j+??ݣ?!?j+??ݣ?R      ??!       Z	?j+??ݣ??j+??ݣ?!?j+??ݣ?JCPU_ONLYY41z??ѱ?b 