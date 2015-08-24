var BasicVis = new function() {

  // If we disable strict, we can use some tricks to
  // give better error messages. By default, we don't
  // do this, to make finding bugs easier and to
  // comply with the style guide.

  'use strict';

  // In non-quirk browser modes, d3.selection.style("attr", int_val)
  // doesn't work! This is because it should be int_val+"px"
  // We wrap around the function to catch this case.

  var style_ = d3.selection.prototype.style;

  d3.selection.prototype.style = function(a,b) {
    if (arguments.length == 1) {
      return style_.call(this, a);
    } else if (typeof b == 'number') {
      style_.call(this, a, b + "px");
    } else {
      style_.call(this, a, b);
    }
    return this;
  };

  // Utilities!
  // =============

  // make_function
  //   A lot of arguments can be a constant
  //   or function. This turns constants into
  //   functions.

  // sometimes d3 requires the data, d, datum to be in function format to work
  var make_function = function(val) {
    if (typeof val == 'function') {
      return val;
    } else {
      return function() {return val;};
    }
  };

  // VisElement
  //   This is a super class for all
  //   our visualization elements

  this.VisElement = function() {

    this.updateTimeout = null;


    // creates methods to reset svg element
    this.layout = function() {};
    this.render = function() {};

    this.update = function() {
      this.layout();
      this.render();
    };

    // methods to reset svg, but with a timer
    this.scheduleUpdate = function(n) {
      var this_ = this;
      n = n || 10;
      var update = function() {
        this_.layout();
        this_.render();
        this_.updateTimeout = null;
      };
      if (this.updateTimeout) clearTimeout(this.updateTimeout);
      this.updateTimeout = setTimeout(update, n);
    };

    // If the window resizes, rebuild or update
    this.bindToWindowResize = function() {
      var this_ = this;
      var scheduleUpdate = function() {
        this_.scheduleUpdate();
      };
      $(window).resize(scheduleUpdate);
    };

    // make_selector
    //   We'll use this in all our constructors
    //   to get a d3 selection to build on

    // complex initialisation of the d3.select("string"), gets returned as make_selector(s)
    this.make_selector = function(s) {
      var caller = '';
      //var caller = arguments.callee.caller.name;
      if (!d3) throw Error(caller + '(): Depends on the D3 library,' +
                            ' which does not seem to be in scope.');
      if (typeof s == 'string') {
        var str = s;
        s = d3.select(s);
        if (s.empty()) throw Error(caller + '(): selector \'' + str +
                          '\' doesn\'t seem to correspond to an element.');
        return s;
      } else if (typeof s == 'object') {
        if ('node' in s) {
          // s seems to be a d3 selector
          return s;
        } else if ('jquery' in s) {
          // s seems to be a jquery object
          throw TypeError(caller + '(): selector can\'t be a JQuery object;' +
                                   ' please use a string or d3.select().');
        }
      }
      throw TypeError(caller + '(): Given selector of type ' + typeof s +
             ' is not a valid selector; please use a string or d3.select().');
    };

  };

  // initialises VisElement as a new object (not inheriting anything)
  this.VisElement.prototype = new Object();



  // Container (which allows upating of a number of children divs)
  //=========================

  this.Container = function Container(s) {
    // this.s = d3.select.(s)
    this.s = this.make_selector(s);
    // this.inner appends a div to the selector (i.e #mnist_graph)
    this.inner = this.s.append('div');
    // create a list of children and children divs - from the DOM?
    this._children = [];
    this._children_divs = [];
    return this;
  };

  // Container inherits from VisElement
  this.Container.prototype = new this.VisElement();

  // constructor might be BasicVis.Scatterplot (i.e - creating a new scatterplot child)
  this.Container.prototype.new_child = function(constructor) {
    //child_div = d3.select(s).append('div').append('div') -> i.e a nested div?
    var child_div = this.inner.append('div');
    // new method: sets the css left and top values to values in passed array v
    child_div.pos = function pos(v) {
      child_div
        .style('left', v[0])
        .style('top', v[1]);
      return child_div;
    };
    // new method: sets the css width and height of the appended div to values in passed array
    child_div.size = function size(v) {
      child_div
        .style('width', v[0])
        .style('height', v[1]);
      return child_div;
    };
    // if passed in function, constructor() == BasicVis.Scatterplot
    // child becomes a new Scatterplot instance
    var child = new constructor(child_div);
    // sets the 'div' of the new object to the same 'div' as the appended child_div
    child.div = child_div;
    // pushes the child_div into the containers list of children 'divs'
    this._children_divs.push(child_div);
    // pushes the child object (i.e - a new scatterplot) into the objects 'child' list
    this._children.push(child);
    // updates the svg based upon new info (calling layout, render, and update)
    this.scheduleUpdate();
    // return the child object, to be used as fit within calling functions 
    return child;
  };

  // overwrites the layout function from VisElement virtual function
  this.Container.prototype.layout = function layout() {
    // parseInt converts the argument to an integer? if a second argument is passed,
    // the number is of that base - i.e: "base64" "base8" = octal
    // in the case of mnist_space - the width of the whole container element (all three boxes)
    var W = parseInt(this.s.style('width'));
    // console.log("Radix in Container Layout: " + W);
    // this.inner is inner-'div', and the second argument passed is the value assigned
    // to the first! i.e position: 'relative'
    this.inner
      .style('width', 1.0 * W)
      .style('position', 'relative');

    if (!this.child_layout)
      throw Error('Container: Must implement child_layout()' +
                  ' to position and size child divs.');

    // goes through the list of children objects (ie ScatterPlots) and changes their
    // style etc
    for (var i = 0; i < this._children.length; i++) {
      this._children_divs[i]
        .style('position', 'absolute');
    }

    // this runs the specific child_layout function created 
    // for the particular object in MnistVis - simply setting the childs style
    this.child_layout();

    // calls the layout function for each of the children
    for (var i = 0; i < this._children.length; i++) {
      this._children[i].layout();
    }
    return this;
  };

  // calls the render function associated with each child object
  // and returns the parent object
  this.Container.prototype.render = function render() {
    for (var i = 0; i < this._children.length; i++) {
      this._children[i].render();
    }
    return this;
  };



  //ImgDisplay
  //================================================================

  this.ImgDisplay = function ImgDisplay(s) {
    // create D3 selector
    this.s = this.make_selector(s);
    // use 'HTML5 canvas' to render the image
    // here appending <canvas> </canvas> tags 
    this.canvas = this.s.append('canvas');
    // create data object
    this._data = {};
    // in d3plus, .shape defines the shape of object to place i: i.e "square"??
    // or simply shape: null; in javascript object
    this._data.shape = null;
    this._data.imgs = null;
  };

  this.ImgDisplay.prototype = new this.VisElement();

  // sets the layout of the Img Display
  this.ImgDisplay.prototype.layout = function layout() {
    // get the size of the width from this
    var W = parseInt(this.s.style('width'));
    // the image-rendering ensures that as images are scaled up they retain their
    // pixel quality, rather than being blended together - i.e: pixelated is new 
    // in chrome 2015
    this.canvas
      .attr('style', 'image-rendering:-moz-crisp-edges;' +
                     'image-rendering: -o-crisp-edges;' +
                     'image-rendering:-webkit-optimize-contrast;' +
                     '-ms-interpolation-mode:nearest-neighbor;' +
                     'image-rendering: pixelated;')
      // .style('border', '1px solid #000000')
      .style("box-shadow", "5px 5px 15px lightgrey")
      .style('width', 1.0 * W)
      .style('height', 1.0 * W);
    // returns with canvas element now defined
    return this;
  };

  this.ImgDisplay.prototype.render = function() {return this;};

  // show here 
  this.ImgDisplay.prototype.show = function(i) {


    var i = parseInt(i);
    // gets the objects imgs and shape values
    var imgs = this._data.imgs;
    var shape = this._data.shape;

    if (shape.length == 2) {
      var X = shape[0];
      var Y = shape[1];
    } else if (shape.length == 3) {
      var X = shape[1];
      var Y = shape[2];
    }

    // getContext(2d) can be used to draw on the canvas
    var ctx = this.canvas[0][0].getContext('2d');
    // getImageData(x-pixels from top left, y-pixels f-t-l, Width, Height) gets a
    // rectangular area of the image data -> below is the full image
    var img = ctx.getImageData(0, 0, X, Y);
    // returns the image data in the format RGBA (red, green, blue, Alpha [opacity])
    // the data is concatenated, so every four array slots are data[0] = red_pixel1,
    // data[1] = red_value_for_pixel2 etc R,G,B,A,R,G,B,A,R,G,B,A etc 
    var imgData = img.data;

    if (!this._data.imgs || !this._data.shape) {
      throw Error('ImgDisplay.show(): Must set ImgDisplay.imgs() ' +
                  'and ImgDisplay.shape() before showing image.');
    }

    // shape of the image: 2D i.e 28*28 -> 1 * 28 * 28
    // given the image is held in a 1D array?
    var imgSize = 1;
    for (var n = 0; n < shape.length; n++) {
      imgSize *= shape[n];
    }

    // constraining bounds of ImageDisplay
    if (imgs.length < imgSize * (i + 1)) {
      throw Error('ImgDisplay.show(): Requested image ' + i +
                  ' out of bounds of ImgDisplay.imgs().');
    }


    if (shape.length == 2) {
      for (var dx = 0; dx < X; ++dx)
      for (var dy = 0; dy < Y; ++dy) {
        // calculating the position in the image
        var pos = dx + shape[0] * dy;
        // set the colour 
        var s = 256 * (1 - imgs[imgSize * i + pos]);
        // index with 4, because of RGBA, set RGBA values
        imgData[4 * pos + 0] = s;
        imgData[4 * pos + 1] = s;
        imgData[4 * pos + 2] = s;
        imgData[4 * pos + 3] = 255;
      }
    } else if (shape.length == 3) {
      for (var c  = 0; c  < 3; ++c )
      for (var dx = 0; dx < X; ++dx)
      for (var dy = 0; dy < Y; ++dy) {
        var pos = dx + shape[1] * dy;
        var s = 256 * (1 - imgs[imgSize * i + pos + shape[1]*shape[2]*c]);
        // sets all values that are RGB to s, and A to 255 (Opaque)
        imgData[4 * pos + ((3-c+2)%3)] = s;
        imgData[4 * pos + 3] = 255;
      }
    }

    // put the image data of the image-data-object back into canvas
    ctx.putImageData(img, 0, 0);

    return this;
  };



  this.ImgDisplay.prototype.imgs = function(val) {
    if (!arguments.length) return this._data.imgs;
    this._data.imgs = val;
    return this;
  };


  // sets the shape (2D / 3D)
  this.ImgDisplay.prototype.shape = function(val) {
    if (!arguments.length) return this._data.shape;

    // sets the html5 canvas height and width
    if (val.length == 2) {
      this.canvas
        .attr('width', val[0])
        .attr('height', val[1]);
    } else {
      this.canvas
        .attr('width', val[1])
        .attr('height', val[2]);
    }
    this._data.shape = val;
    return this;
  };




  // ScatterPlot
  // ============================


  this.ScatterPlot = function ScatterPlot(s) {
    // initialise the d3 selectors
    this.s = this.make_selector(s);
    this.svg = this.s.append('svg');
    // append 'g' which groups objects together in the svg (i.e - like an illustrator
      // group, where when you rotate - you rotate everything together!
    this.zoom_g = this.svg.append('g');

    // set up the data 
    this._data = {};
    this._data.N = 0;
    this._data.scale = 1;
    this._data.color = function() {return 'rgb(50,50,50)';};
    this._data.x = function() {return 0;};
    this._data.y = function() {return 0;};
    this._data.size = function() {return 0;};
    this._data.mouseover = function() {};
    this._data.clicked = function() {};

    // the range of the data, used for scaling linearly
    this._data.xrange = null;
    this._data.yrange = null;

    // assigning the d3.scaling to 
    this.xmap = d3.scale.linear();
    this.ymap = d3.scale.linear();

    var this_ = this;

    // zoom takes the d3.zoom().on("zoom", return this_.zoomed())
    this.zoom = d3.behavior.zoom()
                  .on("zoom", function() {this_._zoomed();});

    // finds min and max values in the data, finds the range, and sets the range
    // with 2% of the range as the border
    this.xrange.fit = function(data) {
      var x1 = d3.min(data);
      var x2 = d3.max(data);
      var dx = x2 - x1;
      // this adds the 2% border on each side
      this_.xrange([x1-0.02*dx, x2+0.02*dx]);
      return this_;
    };

    this.yrange.fit = function(data) {
      var x1 = d3.min(data);
      var x2 = d3.max(data);
      var dx = x2 - x1;
      this_.yrange([x1-0.02*dx, x2+0.02*dx]);
      return this_;
    };


  };

  this.ScatterPlot.prototype = new this.VisElement();


  this.ScatterPlot.prototype.layout = function layout() {
    // gets the width of the parent element?
    var W = parseInt(this.s.style('width'));
    // sets the width and height of the appended svg to W (i.e a square)
    this.svg
      .style('width', W)
      .style('height', W);
    // gets the height of the bounding box
    var H = parseInt(this.s.style('height'));
    // gets the smaller of the two
    var D = Math.min(W, H) / 2 - 2;
    // maps the range, and uses -D because of the upsidedown nature of svg?
    this.xmap.range([W / 2 - D, W / 2 + D]);
    this.ymap.range([H / 2 - D, H / 2 + D]);
    return this;
  };

  //
  this.ScatterPlot.prototype.render = function() {
    // gets the data
    var data = this._data;

    var this_ = this;
    // tells d3 to select all the grouped elements, and create svg 'circle'
    // elements for them for the data in the range specified by N
    var selection = this.zoom_g.selectAll('circle')
                   .data(d3.range(data.N));
    // all the scatterplot points, are the d3 data selection
    this.points = selection;
    // console.log("points: " + this.points);

    // 
    var W = parseInt(this.svg.style('width'));
    var H = parseInt(this.svg.style('height'));
    var D = Math.min(W, H) / 2 - 2;


    // create new circles on svg appending the cirlces to the data in the enter() set
    selection.enter().append('circle')
      .attr('r', 0) // seting radius to 0
      .classed({'highlight' : true}) 
      .on('mouseover', this._data.mouseover)
      .on('click', this._data.clicked); // on mouseover, call the preset function

    // .size() returns number of elements in current section
    // data.scale is calling d3.event.scale() setting the zoom value of the data
    // Math.pow here does data.scale^0.7 (i.e to the power of)
    // size will be used to help reset the size of the circles proportionally
    var size = data.size()/Math.pow(data.scale, 0.7);
    // remove old circles from svg (that are now in the exit() set)
    selection.exit().remove();

    // update/reset circle properties
    // using the d3 transition animation at a rate of 200
    // seting the new circle centres by the d3 linear function
    selection.transition().duration(300)
      .attr('cx', function(d, i) { return this_.xmap(data.x(i)); })
      .attr('cy', function(d, i) { return this_.ymap(data.y(i)); });

    // resize according to filter above
    selection
      .attr('r', size)
      .attr('fill', data.color);

    return this;

  };

  // define N (number of datapoints to filter) with a function...
  this.ScatterPlot.prototype.N = function(val) {
    if (!arguments.length) return this._data.N;
    // once the data has been updated, update and rerender the d3 plots
    this._data.N = val;
    this.scheduleUpdate();
    return this;
  };

  // if colour changes, update everything
  this.ScatterPlot.prototype.color = function(val) {
    if (!arguments.length) return this._data.color;
    this._data.color = make_function(val);
    this.scheduleUpdate();
    return this;
  };

  // if sie changes, update everything
  this.ScatterPlot.prototype.size = function(val) {
    if (!arguments.length) return this._data.size;
    this._data.size = make_function(val);
    this.scheduleUpdate();
    return this;
  };

  // if the x,y value changes, update all
  this.ScatterPlot.prototype.x = function(val) {
    if (!arguments.length) return this._data.x;
    this._data.x = make_function(val);
    this.scheduleUpdate();
    return this;
  };
  this.ScatterPlot.prototype.y = function(val) {
    if (!arguments.length) return this._data.y;
    this._data.y = make_function(val);
    this.scheduleUpdate();
    return this;
  };


  this.ScatterPlot.prototype.xrange = function(val) {
    if (!arguments.length) return this._data.xrange;
    if (!(val.length == 2)) {
      if (val.length > 5)
        throw Error('xrange(): yrange must be an array of length 2.' +
                    ' For example, [-1, 1]. Did you mean to use xrange.fit()?');
      throw Error('xrange(): yrange must be an array of length 2.' +
                 ' For example, [-1, 1].');
    }
    // val must be [min,max] and therefore an array of length 2
    this._data.xrange = val;
    // sets the domain of d3.scale.linear().domain(val) -> i.e the new range
    this.xmap.domain(val);
    this.scheduleUpdate();
    return this;
  };

  this.ScatterPlot.prototype.yrange = function(val) {
    if (!arguments.length) return this._data.yrange;
    if (!(val.length == 2)) {
      if (val.length > 5)
        throw Error('yrange(): yrange must be an array of length 2.' +
                  ' For example, [-1, 1]. Did you mean to use yrange.fit()?');
      throw Error('yrange(): yrange must be an array of length 2.' +
                  ' For example, [-1, 1].');
    }
    this._data.yrange = val;
    this.ymap.domain(val);
    this.scheduleUpdate();
    return this;
  };

  this.ScatterPlot.prototype.mouseover = function(val) {
    if (!arguments.length) return this._data.mouseover;
    // val here is likely to be a function, and will occur when mouseover
    this._data.mouseover = val;
    this.scheduleUpdate();
    return this;
  };


  this.ScatterPlot.prototype.clicked = function(val) {
    if (!arguments.length) return this._data.clicked;
    // val here is likely to be a function, and will occur when mouseover
    this._data.clicked = val;
    this.scheduleUpdate();
    return this;
  };

  // enable zoom gives svg.call(d3.behaviour.zoom)
  this.ScatterPlot.prototype.enable_zoom = function() {
    this.svg.call(this.zoom);
    return this;
  };

  // updates in response to a zoom event - all d3 and in tutorials
  this.ScatterPlot.prototype._zoomed = function() {
    this.zoom_g.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale +")");
    this._data.scale = d3.event.scale;
    this.scheduleUpdate();
  };



  // ToolTip
  // ========================================

  this.Tooltip = function Tooltip() {
    // doesn't select any particular element, just appends to the body!! (i.e whole screen)
    this.div = d3.select('body').append('div')
      .style("position", "absolute");
    // start with tooptips hidden
    this.hide();
    this.timeout = null;
    var this_ = this;
    // when someone hovers, pageX and pageY given the coordinates of your mouse in relation 
    // to the html screen!
    this.div.on("mouseover", function() {
      var pos = [d3.event.pageX + 10, d3.event.pageY + 10 ];
      this_.move(pos);
    });
    return this;
  };

  // tooltips are a javascript object
  this.Tooltip.prototype = new Object();

  // tells the div that contains the tooltip (with relative position) to be 
  // 
  this.Tooltip.prototype.size = function(val) {
    if (!arguments.length) return this.div.style("width");
    this.div.style("width", val);
    return this;
  };

  // tells the the tooltip to go to the val[x,y] - which is the mouse pageX, and pageY
   this.Tooltip.prototype.move = function(val) {
    this.div
      .style("left", val[0])
      .style("top", val[1]);
    return this;
  };

  // unhides the div, and brings it to the front by setting the div's z-index to front
  // this works becuase it's a separate svg to the main file
  this.Tooltip.prototype.unhide = function() {
    this.div
      .style("visibility", "visible")
      .style("z-index", "10");
    //throw Error("just debugging");
    return this;
  };

  // simply hides the div (tooltip)
  this.Tooltip.prototype.hide = function() {
    this.div.style("visibility", "hidden");
    return this;
  };

  // this does all of the above when you move
  this.Tooltip.prototype.bind = function(s, cond) {
    // passing in the selector and the condition
    var this_ = this;
    var timeout = null;
    // called on mouseover and mousemove
    var show = function(i) {
      if (cond && ! cond(i) ) {
        return;
      }
      clearTimeout(timeout);
      this_.timeout = null;
      var pos = [d3.event.pageX + 10, d3.event.pageY + 10 ];
      this_.move(pos);
      this_.display(i);
      this_.unhide();
    };
    s.on("mouseover", show);
    s.on("mousemove", show);
    // linders for 300 milliseconds as it zooms out
    s.on("mouseout", function(i) {
      if (!this_.timeout)
        this_.timeout = setTimeout(function() {this_.hide(); this_.timeout = null;}, 300);
    });
  };

  // just he move part on passing the selector
  this.Tooltip.prototype.bind_move = function(s) {
    var this_ = this;
    s.on("mousemove",  function() { 
      var pos = [d3.event.pageX + 10, d3.event.pageY + 10 ];
      this_.move(pos);
    });
  };

  // ImgTooltip
  //=========================================


  this.ImgTooltip = function ImgTooltip() {
    BasicVis.Tooltip.call(this);
    this.div; // needs the below to do anything
    //  .style("border", "1px solid black");
    // the image tooltip is actually an ImgDisplay object, 
    // that renders on this div
    this.img_display = new BasicVis.ImgDisplay(this.div);
    // sets the size of the div to 40px (as is only a small tooltip)
    this.size("40px");
    return this;
  };

  // creates as an instance of the tooltip
  this.ImgTooltip.prototype = Object.create(this.Tooltip.prototype);

  // as the image is actually getting rendered properly like the graph, 
  // it also needs to be updated 
  this.ImgTooltip.prototype.size = function size(val) {
    if (!arguments.length) return this.div.style("width");
    this.div.style("width", val);
    // update!!
    this.img_display.scheduleUpdate();
    return this;
  };

  // shows the tooltip - standard!
  this.ImgTooltip.prototype.display = function display(i) {
    this.img_display.show(i);
    return this;
  };

  // TextTooltip
  //=========================================

  // creates a new tooltip
  this.TextTooltip = function TextTooltip() {
    // the d3.selection().call function allows you to run this on the tooltip?
    // the .call(this) chains the constructors for TextTooltip to the 
    // constructors below, i.e - the .div etc...a way of creating a more complex object!
    BasicVis.Tooltip.call(this);
    // these are the labels held by the svg graphic?
    this._labels = [];
    // set text properties
    this.div
      .style("background-color", "black")
      .style("color", "white")
      .style("font-size", "20px")
      .style("padding-left", "3px")
      .style("padding-right", "3px")
      .style("padding-top", "2px")
      .style("padding-bottom", "2px")
      .style("box-shadow", "5px 5px 15px lightgrey");
      // .style("border", "1px solid black");
    // return the new tooltip
    return this;
  };

  // creates it as an instance of the tooltip
  this.TextTooltip.prototype = Object.create(this.Tooltip.prototype);

  // display is used by the tooltip function when updating on,
  // and attaches the correct label to the tooltip before unhiding!
  this.TextTooltip.prototype.display = function display(i) {
    // i is the index of the label
    var labels = this._labels;
    // this is the text passed for the div (if it exists)
    if (i < labels.length){
      this.div.text(labels[i]);
    } else {
      this.div.text("");
    }
    return this;
  };


};
