
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
      if(this.away_team === this.home_team){
        alert('Cannot compare same teams');
        return;
      }
      axios.post('/__predict', {
        away_team: this.away_team,
        home_team: this.home_team
      })
      .then(function (response) {
        Window.App.result = response.data.response.data.result;
      })
      .catch(function (error) {
        console.log(error);
      });
    }
  },
  computed : {
    modalOpen : function(){
      return this.result.length <= 0;
    },
    winner: function(){
      if(this.result === 'A') return this.away_team+" is likely to win";
      if(this.result === 'H') return this.home_team+" is likely to win";
      if(this.result === 'D') return "Match is likely to be a draw ";
    },
    winnerletter : function(){
      if(this.result === 'A' || this.result === 'H') return "W";
      if(this.result === 'D') return "D";
    },
    winnerclass : function(){
      if(this.result === 'A' || this.result === 'H') return "win";
      if(this.result === 'D') return "draw";
    }
  },
  created : function(){
    Window.App = this;
  }
});