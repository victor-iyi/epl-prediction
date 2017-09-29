new Vue({
  el : '.page-container',
  data :  {
    result : ''
  },
  methods : {
    open : function(){
      this.result = '';
    }
  },
  computed : {
    modalOpen : function(){
      return this.result.length <= 0;
    }
  }
});