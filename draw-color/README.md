# draw-color
A Tensorflow implementation of [DRAW](https://arxiv.org/abs/1502.04623). Now includes support for colored images!

This is code to accompany [my post on the DRAW model](http://kvfrans.com/what-is-draw-deep-recurrent-attentive-writer/).

For an explanation of how I modified the original DRAW into a colored model, check out [my post on colorizing DRAW](http://kvfrans.com/colorizing-the-draw-model/).

<img src="http://kvfrans.com/content/images/2016/10/res_noattn.gif">
<img src="http://kvfrans.com/content/images/2016/10/res_attention.gif">

<img src="http://kvfrans.com/content/images/2016/12/output_Y5Da6R.gif">
<img src="http://kvfrans.com/content/images/2016/12/output_ckIGZC.gif">

The original DRAW is a slightly rewritten, commented version of [ericjang's implementation](https://github.com/ericjang/draw), running on MNIST.

Usage:
```
python main.py
```

The colored DRAW runs on the celebA dataset.

Usage:
```
python color.py
```
