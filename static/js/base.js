
new Vue({
  el : '.page-container',
  data :  {
    result : '',
    home_team: '',
    away_team: ''
  },
  methods : {
    open : function(){
      this.result = '';
    },
    predict : function(){
      if(this.away_team == this.home_team){
        alert('Cannot compare');
      }
      axios.post('/__predict', {
        away_team: this.away_team,
        home_team: this.home_team
      })
      .then(function (response) {
        console.log(response);
      })
      .catch(function (error) {
        console.log(error);
      });
    }
  },
  computed : {
    modalOpen : function(){
      return this.result.length <= 0;
    }
  },
  created : function(){
    Window.App = this;
  }
});