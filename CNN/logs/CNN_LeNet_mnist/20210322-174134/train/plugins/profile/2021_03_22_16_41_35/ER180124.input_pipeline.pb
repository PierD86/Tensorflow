	\ A?c?O@\ A?c?O@!\ A?c?O@	k?}????k?}????!k?}????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$\ A?c?O@??_?L??Ax??#??O@Y??ׁsF??*	?????N@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ZӼ???!a|<??@@)?5?;Nё?1??y???<@:Preprocessing2F
Iterator::Model?z6?>??!?Mq???B@)vq?-??1?%?J:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??_vO??!?*?0U?1@)?ZӼ?}?1?J?'@:Preprocessing2U
Iterator::Model::ParallelMapV2lxz?,C|?!?{?3?&@)lxz?,C|?1?{?3?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???&??!??dO@)a2U0*?s?1H/??^?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???_vOn?!rP?(?@)???_vOn?1rP?(?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!???{?@)?~j?t?h?1???{?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?HP???!V&?L4@)Ǻ???V?1?[Q)??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9k?}????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??_?L????_?L??!??_?L??      ??!       "      ??!       *      ??!       2	x??#??O@x??#??O@!x??#??O@:      ??!       B      ??!       J	??ׁsF????ׁsF??!??ׁsF??R      ??!       Z	??ׁsF????ׁsF??!??ׁsF??JCPU_ONLYYk?}????b 