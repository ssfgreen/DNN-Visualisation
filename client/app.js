
// creating an angular app, which uses ngResource (makes RESTful API stuff work better)
var app = angular.module('NeuralNet',['ngResource','ngStorage']);

// a factory has globally avaliable scope? whereas the controller has local scope
app.factory('Result',['$resource',Result]);
    
// this sets up all of our RESTful calls -> 
function Result($resource){
    return $resource('/results/:resultId');
};

// this is our controller, that uses the resource RESTful stuff, the controller
app.controller('NeuralController',['$scope','Result','$localStorage',NeuralController]);

// this is our Controller that manages 
function NeuralController($scope,Result,$localStorage){

    // in the index.html I can access the result variable, which gets 
    // right thing in response to the passed ID???
    // $scope.result = Result.get({resultId:'id'});

    // the function that upon change uploads the data to the local storage
    $scope.getID = function(id){
        $scope.result = Result.get({resultId:id});
        $scope.result.$promise.then(function(){
            console.log($scope.result);
            $localStorage.result = $scope.result.data;
        });
    }

    // watches for the result_id to change in the angular list
    $scope.$watch('result_id',function(new_id){
        $scope.getID(new_id);
    });

    // this gets all the results?
    $scope.results = Result.query();

    // returns a results promise? callback? when waiting for server to respond
    $scope.results.$promise.then(function(){
            console.log('loaded');
            console.log($scope.results);
        },function(error){
            console.log(error);
        })

    // creates a new Result object?
    var result = new Result();

    // 
    // the function that upon change uploads the data to the local storage
    $scope.runNet = function(id){
        result.name = id;
        result.$save().then(function(){
            console.log('saved');
        },function(error){
            console.log(error);
        })
    }

    // result.name = 'hello';
    // result.$save().then(function(){
    //     console.log('saved');
    // },function(error){
    //     console.log(error);
    // })
};


app.controller('VisController',['$scope','Result','$localStorage',VisController]);

function VisController($scope,Result,$localStorage){

    $scope.coords = $localStorage.result.DATA.COORDS;

    $scope.tsne = $localStorage.result.DATA.TSNE_DATA;

}