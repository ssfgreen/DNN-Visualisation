 """ Adapted from Chris Olahs' blog post on Visualising MNIST """

  // UpdateClass
  //   This is a super class for all
  //   our visualization elements

  var UpdateClass = function() {

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
      var that = this;
      n = n || 10;
      var update = function() {
        that.layout();
        that.render();
        that.updateTimeout = null;
      };
      if (this.updateTimeout) clearTimeout(this.updateTimeout);
      this.updateTimeout = setTimeout(update, n);
    };

    // If the window resizes, rebuild or update
    this.bindToWindowResize = function() {
      var that = this;
      var scheduleUpdate = function() {
        that.scheduleUpdate();
      };
      $(window).resize(scheduleUpdate);
    };

    // d3Selector
    //   We'll use this in all our constructors
    //   to get a d3 selection to build on - taken from Chris Olah

    // complex initialisation of the d3.select("string"), gets returned as d3Selector(s)
    this.d3Selector = function(s) {
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

  // initialises UpdateClass as a new object (not inheriting anything)
  UpdateClass.prototype = new Object();








  // GroupingClass (which allows upating of a number of children divs)
  //=========================

  var GroupingClass = function GroupingClass(s) {
    // this.s = d3.select.(s)
    this.s = this.d3Selector(s);
    // this.nestedDiv appends a div to the selector (i.e #mnist_graph)
    this.nestedDiv = this.s.append('div');
    // create a list of children and children divs 
    this.classChildren = [];
    this.classChildrenDivs = [];
    return this;
  };

  // GroupingClass inherits from UpdateClass
  GroupingClass.prototype = new UpdateClass();

  // constructor might be Updating.ScatterPlotClass (i.e - creating a new ScatterPlotClass child)
  GroupingClass.prototype.CreateChild = function(constructor) {
    //childDiv = d3.select(s).append('div').append('div') -> i.e a nested div?
    var childDiv = this.nestedDiv.append('div');
    // new method: sets the css left and top values to values in passed array v
    childDiv.pos = function pos(v) {
      childDiv
        .style('left', v[0])
        .style('top', v[1]);
      return childDiv;
    };
    // new method: sets the css width and height of the appended div to values in passed array
    childDiv.size = function size(v) {
      childDiv
        .style('width', v[0])
        .style('height', v[1]);
      return childDiv;
    };
    // if passed in function, constructor() == Updating.ScatterPlotClass
    // child becomes a new ScatterPlotClass instance
    var child = new constructor(childDiv);
    // sets the 'div' of the new object to the same 'div' as the appended childDiv
    child.div = childDiv;
    // pushes the childDiv into the GroupingClasss list of children 'divs'
    this.classChildrenDivs.push(childDiv);
    // pushes the child object (i.e - a new ScatterPlotClass) into the objects 'child' list
    this.classChildren.push(child);
    // updates the svg based upon new info (calling layout, render, and update)
    this.scheduleUpdate();
    // return the child object, to be used as fit within calling functions 
    return child;
  };

  // overwrites the layout function from UpdateClass virtual function
  GroupingClass.prototype.layout = function layout() {
    // parseInt converts the argument to an integer? if a second argument is passed,
    // the number is of that base - i.e: "base64" "base8" = octal
    // in the case of mnist_space - the width of the whole GroupingClass element (all three boxes)
    var W = parseInt(this.s.style('width'));
    // console.log("Radix in GroupingClass Layout: " + W);
    // this.nestedDiv is nestedDiv-'div', and the second argument passed is the value assigned
    // to the first! i.e position: 'relative'
    this.nestedDiv
      .style('width', 1.0 * W)
      .style('position', 'relative');

    if (!this.childLayout)
      throw Error('GroupingClass: Must implement childLayout()' +
                  ' to position and size child divs.');

    // goes through the list of children objects (ie ScatterPlotClasss) and changes their
    // style etc
    for (var i = 0; i < this.classChildren.length; i++) {
      this.classChildrenDivs[i]
        .style('position', 'absolute');
    }

    // this runs the specific childLayout function created 
    // for the particular object in MnistVis - simply setting the childs style
    this.childLayout();

    // calls the layout function for each of the children
    for (var i = 0; i < this.classChildren.length; i++) {
      this.classChildren[i].layout();
    }
    return this;
  };

  // calls the render function associated with each child object
  // and returns the parent object
  GroupingClass.prototype.render = function render() {
    for (var i = 0; i < this.classChildren.length; i++) {
      this.classChildren[i].render();
    }
    return this;
  };








  // ScatterPlotClass
  // ============================


  var ScatterPlotClass = function ScatterPlotClass(s) {
    // initialise the d3 selectors
    this.s = this.d3Selector(s);
    this.svg = this.s.append('svg');
    // append 'g' which groups objects together in the svg (i.e - like an illustrator
      // group, where when you rotate - you rotate everything together!
    this.zoom_g = this.svg.append('g');

    // set up the data 
    this.dataObj = {};
    this.dataObj.N = 0;
    this.dataObj.scale = 1;
    this.dataObj.color = function() {return 'rgb(50,50,50)';};
    this.dataObj.x = function() {return 0;};
    this.dataObj.y = function() {return 0;};
    this.dataObj.size = function() {return 0;};
    this.dataObj.mouseover = function() {};
    this.dataObj.clicked = function() {};

    // the range of the data, used for scaling linearly
    this.dataObj.xrange = null;
    this.dataObj.yrange = null;

    // assigning the d3.scaling to 
    this.d3ScaleLinearX = d3.scale.linear();
    this.d3ScaleLinearY = d3.scale.linear();

    var that = this;

    // zoom takes the d3.zoom().on("zoom", return that.zoomed())
    this.zoom = d3.behavior.zoom()
                  .on("zoom", function() {that._zoomed();});

    // finds min and max values in the data, finds the range, and sets the range
    // with 2% of the range as the border
    this.xrange.fit = function(data) {
      var x1 = d3.min(data);
      var x2 = d3.max(data);
      var dx = x2 - x1;
      // this adds the 2% border on each side
      that.xrange([x1-0.02*dx, x2+0.02*dx]);
      return that;
    };

    this.yrange.fit = function(data) {
      var x1 = d3.min(data);
      var x2 = d3.max(data);
      var dx = x2 - x1;
      that.yrange([x1-0.02*dx, x2+0.02*dx]);
      return that;
    };


  };

  ScatterPlotClass.prototype = new UpdateClass();


  ScatterPlotClass.prototype.layout = function layout() {
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
    this.d3ScaleLinearX.range([W / 2 - D, W / 2 + D]);
    this.d3ScaleLinearY.range([H / 2 - D, H / 2 + D]);
    return this;
  };

  //
  ScatterPlotClass.prototype.render = function() {
    // gets the data
    var data = this.dataObj;

    var that = this;
    // tells d3 to select all the grouped elements, and create svg 'circle'
    // elements for them for the data in the range specified by N
    var selection = this.zoom_g.selectAll('circle')
                   .data(d3.range(data.N));
    // all the ScatterPlotClass points, are the d3 data selection
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
      .on('mouseover', this.dataObj.mouseover)
      .on('click', this.dataObj.clicked); // on mouseover, call the preset function

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
      .attr('cx', function(d, i) { return that.d3ScaleLinearX(data.x(i));})
      .attr('cy', function(d, i) { return that.d3ScaleLinearY(data.y(i));});

    // resize according to filter above
    selection
      .attr('r', size)
      .attr('fill', data.color);

    return this;

  };

  // define N (number of datapoints to filter) with a function...
  ScatterPlotClass.prototype.N = function(val) {
    if (!arguments.length) return this.dataObj.N;
    // once the data has been updated, update and rerender the d3 plots
    this.dataObj.N = val;
    this.scheduleUpdate();
    return this;
  };

  // if colour changes, update everything
  ScatterPlotClass.prototype.color = function(val) {
    if (!arguments.length) return this.dataObj.color;
    this.dataObj.color = make_function(val);
    this.scheduleUpdate();
    return this;
  };

  // if sie changes, update everything
  ScatterPlotClass.prototype.size = function(val) {
    if (!arguments.length) return this.dataObj.size;
    this.dataObj.size = make_function(val);
    this.scheduleUpdate();
    return this;
  };

  // if the x,y value changes, update all
  ScatterPlotClass.prototype.x = function(val) {
    if (!arguments.length) return this.dataObj.x;
    this.dataObj.x = make_function(val);
    this.scheduleUpdate();
    return this;
  };
  ScatterPlotClass.prototype.y = function(val) {
    if (!arguments.length) return this.dataObj.y;
    this.dataObj.y = make_function(val);
    this.scheduleUpdate();
    return this;
  };


  ScatterPlotClass.prototype.xrange = function(val) {
    if (!arguments.length) return this.dataObj.xrange;
    if (!(val.length == 2)) {
      if (val.length > 5)
        throw Error('xrange(): yrange must be an array of length 2.' +
                    ' For example, [-1, 1]. Did you mean to use xrange.fit()?');
      throw Error('xrange(): yrange must be an array of length 2.' +
                 ' For example, [-1, 1].');
    }
    // val must be [min,max] and therefore an array of length 2
    this.dataObj.xrange = val;
    // sets the domain of d3.scale.linear().domain(val) -> i.e the new range
    this.d3ScaleLinearX.domain(val);
    this.scheduleUpdate();
    return this;
  };

  ScatterPlotClass.prototype.yrange = function(val) {
    if (!arguments.length) return this.dataObj.yrange;
    if (!(val.length == 2)) {
      if (val.length > 5)
        throw Error('yrange(): yrange must be an array of length 2.' +
                  ' For example, [-1, 1]. Did you mean to use yrange.fit()?');
      throw Error('yrange(): yrange must be an array of length 2.' +
                  ' For example, [-1, 1].');
    }
    this.dataObj.yrange = val;
    this.d3ScaleLinearY.domain(val);
    this.scheduleUpdate();
    return this;
  };

  ScatterPlotClass.prototype.mouseover = function(val) {
    if (!arguments.length) return this.dataObj.mouseover;
    // val here is likely to be a function, and will occur when mouseover
    this.dataObj.mouseover = val;
    this.scheduleUpdate();
    return this;
  };


  ScatterPlotClass.prototype.clicked = function(val) {
    if (!arguments.length) return this.dataObj.clicked;
    // val here is likely to be a function, and will occur when mouseover
    this.dataObj.clicked = val;
    this.scheduleUpdate();
    return this;
  };

  // enable zoom gives svg.call(d3.behaviour.zoom)
  ScatterPlotClass.prototype.enable_zoom = function() {
    this.svg.call(this.zoom);
    return this;
  };

  // updates in response to a zoom event - all d3 and in tutorials
  ScatterPlotClass.prototype._zoomed = function() {
    this.zoom_g.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale +")");
    this.dataObj.scale = d3.event.scale;
    this.scheduleUpdate();
  };







  // ImageClass
  //================================================================

  var ImageClass = function ImageClass(s) {
    // create D3 selector
    this.s = this.d3Selector(s);
    // use 'HTML5 canvas' to render the image
    // here appending <canvas> </canvas> tags 
    this.canvas = this.s.append('canvas');
    // create data object
    this.dataObj = {};
    // in d3plus, .shape defines the shape of object to place i: i.e "square"??
    // or simply shape: null; in javascript object
    this.dataObj.shape = null;
    this.dataObj.imgs = null;
  };

  ImageClass.prototype = new UpdateClass();

  // sets the layout of the Img Display
  ImageClass.prototype.layout = function layout() {
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

  ImageClass.prototype.render = function() {return this;};

  // show here 
  ImageClass.prototype.show = function(i) {


    var i = parseInt(i);
    // gets the objects imgs and shape values
    var imgs = this.dataObj.imgs;
    var shape = this.dataObj.shape;

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

    if (!this.dataObj.imgs || !this.dataObj.shape) {
      throw Error('ImageClass.show(): Must set ImageClass.imgs() ' +
                  'and ImageClass.shape() before showing image.');
    }

    // shape of the image: 2D i.e 28*28 -> 1 * 28 * 28
    // given the image is held in a 1D array?
    var imgSize = 1;
    for (var n = 0; n < shape.length; n++) {
      imgSize *= shape[n];
    }

    // constraining bounds of ImageDisplay
    if (imgs.length < imgSize * (i + 1)) {
      throw Error('ImageClass.show(): Requested image ' + i +
                  ' out of bounds of ImageClass.imgs().');
    }


    if (shape.length == 2) {
      // indexing the data-array passed in as an array (i.e - each intensity 
      // value from the MNIST dataset is passed in and set a value on the canvas
      // correspondingly)
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



  ImageClass.prototype.imgs = function(val) {
    if (!arguments.length) return this.dataObj.imgs;
    this.dataObj.imgs = val;
    return this;
  };


  // sets the shape (2D / 3D)
  ImageClass.prototype.shape = function(val) {
    if (!arguments.length) return this.dataObj.shape;

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
    this.dataObj.shape = val;
    return this;
  };


  // TooltipClass
  // ========================================

  var TooltipClass = function TooltipClass() {
    // doesn't select any particular element, just appends to the body!! (i.e whole screen)
    this.div = d3.select('body').append('div')
      .style("position", "absolute");
    // start with tooptips hidden
    this.hide();
    this.timeout = null;
    var that = this;
    // when someone hovers, pageX and pageY given the coordinates of your mouse in relation 
    // to the html screen!
    this.div.on("mouseover", function() {
      var pos = [d3.event.pageX + 10, d3.event.pageY + 10 ];
      that.move(pos);
    });
    return this;
  };

  // TooltipClasss are a javascript object
  TooltipClass.prototype = new Object();

  // tells the div that contains the TooltipClass (with relative position) to be 
  // 
  TooltipClass.prototype.size = function(val) {
    if (!arguments.length) return this.div.style("width");
    this.div.style("width", val);
    return this;
  };

  // tells the the TooltipClass to go to the val[x,y] - which is the mouse pageX, and pageY
   TooltipClass.prototype.move = function(val) {
    this.div
      .style("left", val[0])
      .style("top", val[1]);
    return this;
  };

  // unhides the div, and brings it to the front by setting the div's z-index to front
  // this works becuase it's a separate svg to the main file
  TooltipClass.prototype.unhide = function() {
    this.div
      .style("visibility", "visible")
      .style("z-index", "10");
    //throw Error("just debugging");
    return this;
  };

  // simply hides the div (TooltipClass)
  TooltipClass.prototype.hide = function() {
    this.div.style("visibility", "hidden");
    return this;
  };

  // this does all of the above when you move
  TooltipClass.prototype.bind = function(s, cond) {
    // passing in the selector and the condition
    var that = this;
    var timeout = null;
    // called on mouseover and mousemove
    var show = function(i) {
      if (cond && ! cond(i) ) {
        return;
      }
      clearTimeout(timeout);
      that.timeout = null;
      var pos = [d3.event.pageX + 10, d3.event.pageY + 10 ];
      that.move(pos);
      that.display(i);
      that.unhide();
    };
    s.on("mouseover", show);
    s.on("mousemove", show);
    // linders for 300 milliseconds as it zooms out
    s.on("mouseout", function(i) {
      if (!that.timeout)
        that.timeout = setTimeout(function() {that.hide(); that.timeout = null;}, 300);
    });
  };

  // just he move part on passing the selector
  TooltipClass.prototype.bind_move = function(s) {
    var that = this;
    s.on("mousemove",  function() { 
      var pos = [d3.event.pageX + 10, d3.event.pageY + 10 ];
      that.move(pos);
    });
  };

  // ImageTooltipClass
  //=========================================


  var ImageTooltipClass = function ImageTooltipClass() {
    TooltipClass.call(this);
    this.div; // needs the below to do anything
    //  .style("border", "1px solid black");
    // the image TooltipClass is actually an ImageClass object, 
    // that renders on this div
    this.img_display = new ImageClass(this.div);
    // sets the size of the div to 40px (as is only a small TooltipClass)
    this.size("40px");
    return this;
  };

  // creates as an instance of the TooltipClass
  ImageTooltipClass.prototype = Object.create(this.TooltipClass.prototype);

  // as the image is actually getting rendered properly like the graph, 
  // it also needs to be updated 
  ImageTooltipClass.prototype.size = function size(val) {
    if (!arguments.length) return this.div.style("width");
    this.div.style("width", val);
    // update!!
    this.img_display.scheduleUpdate();
    return this;
  };

  // shows the TooltipClass - standard!
  ImageTooltipClass.prototype.display = function display(i) {
    this.img_display.show(i);
    return this;
  };

  // TextTooltipClass
  //=========================================

  // creates a new TooltipClass
  var TextTooltipClass = function TextTooltipClass() {
    // the d3.selection().call function allows you to run this on the TooltipClass?
    // the .call(this) chains the constructors for TextTooltipClass to the 
    // constructors below, i.e - the .div etc...a way of creating a more complex object!
    TooltipClass.call(this);
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
    // return the new TooltipClass
    return this;
  };

  // creates it as an instance of the TooltipClass
  TextTooltipClass.prototype = Object.create(this.TooltipClass.prototype);

  // display is used by the TooltipClass function when updating on,
  // and attaches the correct label to the TooltipClass before unhiding!
  TextTooltipClass.prototype.display = function display(i) {
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


// -- HELPERS

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


//----> The Visualisation Plot
// sometimes dataX and dataY are the same

var metaVis = function metaVis(selector, imageArr,
              ControlX, ControlY,
              ControlN, ControlCol,
              dataLabels, tsneData,
              sensible_number_of_points) {

  // initialise as a GroupingClass with children
  GroupingClass.call(this, selector);

  // appends a div to use later to print the info about the net (params etc)
  // initialise GroupingClass variables
  this.info_meta = this.nestedDiv.append("div");
  this.info_vis = this.nestedDiv.append("div");
  this.imageArr = imageArr;
  this.ControlX = ControlX;
  this.ControlY = ControlY;
  this.ControlN = ControlN;
  this.ControlCol = ControlCol;
  this.dataLabels = dataLabels;
  this.tsneData = tsneData;
  this.sensible_number_of_points = sensible_number_of_points;

  that = this;

  // set TooltipClasss function
  function TooltipClasss (tooltype, appendage, labels) {
    // console.log("called");
    if (tooltype == "text") {
      console.log(tooltype);
      // creates a new TooltipClass, and binds to the new data!
      this.tooltext = new TextTooltipClass();
      this.tooltext._labels = labels;
      this.tooltext.bind(appendage.points);
      this.tooltext.bind_move(appendage.s);
      this.tooltext.div.style("font-size", "85%");
    } else if (tooltype == "img") {
      console.log(tooltype);
      this.toolimg = new ImageTooltipClass();
      this.toolimg.img_display.shape([28,28]);
      this.toolimg.img_display.imgs(that.imageArr); // replace with mnist_images
      this.toolimg.bind(appendage.points);
      this.toolimg.bind_move(appendage.s);
    };
  }

  this.meta_display = this.CreateChild(ScatterPlotClass)
    .N(this.ControlN) //DATA.META.length/2
    .xrange.fit(that.ControlX)
    .yrange.fit(that.ControlY)
    .x(function(i) {return that.ControlY == that.ControlX ? that.ControlX[2*i] : that.ControlX[i];})
    .y(function(i) {return that.ControlY == that.ControlX ? that.ControlY[2*i+1] : that.ControlY[i];})
    .color(function(i) {
      // set according to the label
      var hue = 180;
      var saturation = that.ControlCol[i]/10.0;//DATA.LAYER[i]/10.0;
      var lightness = 0.5;
      return d3.hsl(hue, saturation, lightness);
    });

  this.meta_display
    .enable_zoom()
    .bindToWindowResize()

  this.main_display = this.CreateChild(ScatterPlotClass)
    .N(0)
    .color( function(i) {
      var hue = 180*that.dataLabels[i]/10.0;
      var saturation = 0.5;
      var lightness = 0.5;
      return d3.hsl(hue, saturation, lightness);
    });

  var main_display = this.main_display;

  this.main_display.show = function show(i) {

    var coords = that.tsneData[i];
    var labels = that.dataLabels;

    var NO_POINTS = 0;

    if (that.sensible_number_of_points < that.tsneData.length/2) {
      NO_POINTS = that.sensible_number_of_points;
    } else {
      NO_POINTS = that.tsneData[i].length/2;
    }

    main_display
      .N(NO_POINTS)
      .xrange.fit(coords)
      .yrange.fit(coords)
      .x(function(i) {return coords[2*i  ];}) // gets the x-coords responding to the index
      .y(function(i) {return coords[2*i+1];});

    main_display
    .enable_zoom()
    .bindToWindowResize()

    TooltipClasss("img", main_display, labels);
  }

  // javascript scoping, to ensure it is THIS function
  // not the this of the 'circle' i.e
  var that = this;

  this.meta_display.mouseover( function(i) {

    that.info_meta.html("<center> Layer: " + DATA.LAYER[i] + " Epoch: " + DATA.EPOCH[i]
        + "</center");
    
    that.main_display.show(i);
  });

}

metaVis.prototype = Object.create(GroupingClass.prototype);

metaVis.prototype.childLayout = function childLayout() {
 // gets the width of the main representation div 
  W = parseInt(this.s.style('width'));

  var gutter = W/16;
  var main = W - gutter*6;
  var main_box_size = (main/3) * 2;
  var nav_box_size = main/3;
  var dims = [main_box_size, main_box_size];
  var dims_nav = [nav_box_size, nav_box_size];
  var H = W/2;

  this.nestedDiv
    .style('width', W)
    .style('height', H); 

  this.meta_display.size(W/300);
  this.meta_display.div
    // .style('border', '1px solid #959595')
    .style('background-color', 'white')
    .pos([gutter*2,H/3])
    .size(dims_nav); 

  this.main_display.size(W/400);
  this.main_display.div
    // .style('border', '1px solid #959595')
    .style('background-color', 'white')
    .pos([nav_box_size+gutter*4,gutter])
    .size(dims); 

  this.info_meta
    .style('position', 'absolute')
    .style('font-size', '14px')
    .style('width', W)
    .style('height', box_size/3)
    .style('top', gutter)
    .style("fill", "red");
    // .style('color', 'blue');
    // .style('left', gutter*3)
    // .style('text-align', center);

  return this;
}
