// Generated by CoffeeScript 1.12.3

/*
A module for DELPHI's Epidata API.

https://github.com/cmu-delphi/delphi-epidata

Notes:
 - If running in node.js (or using browserify), there are no external
   dependencies. Otherwise, a global jQuery object named `$` must be available.
 */

(function() {
  var Epidata, https, querystring;

  if ((typeof $ !== "undefined" && $ !== null ? $.getJSON : void 0) == null) {
    https = require('https');
    querystring = require('querystring');
  }

  Epidata = (function() {
    var BASE_URL, _list, _listitem, _request;

    function Epidata() {}

    BASE_URL = 'https://delphi.midas.cs.cmu.edu/epidata/api.php';

    _listitem = function(value) {
      if (value.hasOwnProperty('from') && value.hasOwnProperty('to')) {
        return value['from'] + "-" + value['to'];
      } else {
        return "" + value;
      }
    };

    _list = function(values) {
      var value;
      if (!Array.isArray(values)) {
        values = [values];
      }
      return ((function() {
        var i, len, results;
        results = [];
        for (i = 0, len = values.length; i < len; i++) {
          value = values[i];
          results.push(_listitem(value));
        }
        return results;
      })()).join(',');
    };

    _request = function(callback, params) {
      var handler, reader;
      handler = function(data) {
        if ((data != null ? data.result : void 0) != null) {
          return callback(data.result, data.message, data.epidata);
        } else {
          return callback(0, 'unknown error', null);
        }
      };
      if ((typeof $ !== "undefined" && $ !== null ? $.getJSON : void 0) != null) {
        return $.getJSON(BASE_URL, params, handler);
      } else {
        reader = function(response) {
          var text;
          text = '';
          response.setEncoding('utf8');
          response.on('data', function(chunk) {
            return text += chunk;
          });
          response.on('error', function(e) {
            return error(e.message);
          });
          return response.on('end', function() {
            return handler(JSON.parse(text));
          });
        };
        return https.get(BASE_URL + "?" + (querystring.stringify(params)), reader);
      }
    };

    Epidata.range = function(from, to) {
      var ref;
      if (to <= from) {
        ref = [to, from], from = ref[0], to = ref[1];
      }
      return {
        from: from,
        to: to
      };
    };

    Epidata.fluview = function(callback, regions, epiweeks, issues, lag, auth) {
      var params;
      if (!((regions != null) && (epiweeks != null))) {
        throw {
          msg: '`regions` and `epiweeks` are both required'
        };
      }
      if ((issues != null) && (lag != null)) {
        throw {
          msg: '`issues` and `lag` are mutually exclusive'
        };
      }
      params = {
        'source': 'fluview',
        'regions': _list(regions),
        'epiweeks': _list(epiweeks)
      };
      if (issues != null) {
        params.issues = _list(issues);
      }
      if (lag != null) {
        params.lag = lag;
      }
      if (auth != null) {
        params.auth = auth;
      }
      return _request(callback, params);
    };

    Epidata.fluview_clinical = function(callback, regions, epiweeks, issues, lag) {
      var params;
      if (!((regions != null) && (epiweeks != null))) {
        throw {
          msg: '`regions` and `epiweeks` are both required'
        };
      }
      if ((issues != null) && (lag != null)) {
        throw {
          msg: '`issues` and `lag` are mutually exclusive'
        };
      }
      params = {
        'source': 'fluview_clinical',
        'regions': _list(regions),
        'epiweeks': _list(epiweeks)
      };
      if (issues != null) {
        params.issues = _list(issues);
      }
      if (lag != null) {
        params.lag = lag;
      }
      return _request(callback, params);
    };

    Epidata.flusurv = function(callback, locations, epiweeks, issues, lag) {
      var params;
      if (!((locations != null) && (epiweeks != null))) {
        throw {
          msg: '`locations` and `epiweeks` are both required'
        };
      }
      if ((issues != null) && (lag != null)) {
        throw {
          msg: '`issues` and `lag` are mutually exclusive'
        };
      }
      params = {
        'source': 'flusurv',
        'locations': _list(locations),
        'epiweeks': _list(epiweeks)
      };
      if (issues != null) {
        params.issues = _list(issues);
      }
      if (lag != null) {
        params.lag = lag;
      }
      return _request(callback, params);
    };

    Epidata.gft = function(callback, locations, epiweeks) {
      var params;
      if (!((locations != null) && (epiweeks != null))) {
        throw {
          msg: '`locations` and `epiweeks` are both required'
        };
      }
      params = {
        'source': 'gft',
        'locations': _list(locations),
        'epiweeks': _list(epiweeks)
      };
      return _request(callback, params);
    };

    Epidata.ght = function(callback, auth, locations, epiweeks, query) {
      var params;
      if (!((auth != null) && (locations != null) && (epiweeks != null) && (query != null))) {
        throw {
          msg: '`auth`, `locations`, `epiweeks`, and `query` are all required'
        };
      }
      params = {
        'source': 'ght',
        'auth': auth,
        'locations': _list(locations),
        'epiweeks': _list(epiweeks),
        'query': query
      };
      return _request(callback, params);
    };

    Epidata.twitter = function(callback, auth, locations, dates, epiweeks) {
      var params;
      if (!((auth != null) && (locations != null))) {
        throw {
          msg: '`auth` and `locations` are both required'
        };
      }
      if (!((dates != null) ^ (epiweeks != null))) {
        throw {
          msg: 'exactly one of `dates` and `epiweeks` is required'
        };
      }
      params = {
        'source': 'twitter',
        'auth': auth,
        'locations': _list(locations)
      };
      if (dates != null) {
        params.dates = _list(dates);
      }
      if (epiweeks != null) {
        params.epiweeks = _list(epiweeks);
      }
      return _request(callback, params);
    };

    Epidata.wiki = function(callback, articles, dates, epiweeks, hours) {
      var params;
      if (articles == null) {
        throw {
          msg: '`articles` is required'
        };
      }
      if (!((dates != null) ^ (epiweeks != null))) {
        throw {
          msg: 'exactly one of `dates` and `epiweeks` is required'
        };
      }
      params = {
        'source': 'wiki',
        'articles': _list(articles)
      };
      if (dates != null) {
        params.dates = _list(dates);
      }
      if (epiweeks != null) {
        params.epiweeks = _list(epiweeks);
      }
      if (hours != null) {
        params.hours = _list(hours);
      }
      return _request(callback, params);
    };

    Epidata.cdc = function(callback, auth, epiweeks, locations) {
      var params;
      if (!((auth != null) && (epiweeks != null) && (locations != null))) {
        throw {
          msg: '`auth`, `epiweeks`, and `locations` are all required'
        };
      }
      params = {
        'source': 'cdc',
        'auth': auth,
        'epiweeks': _list(epiweeks),
        'locations': _list(locations)
      };
      return _request(callback, params);
    };

    Epidata.quidel = function(callback, auth, epiweeks, locations) {
      var params;
      if (!((auth != null) && (epiweeks != null) && (locations != null))) {
        throw {
          msg: '`auth`, `epiweeks`, and `locations` are all required'
        };
      }
      params = {
        'source': 'quidel',
        'auth': auth,
        'epiweeks': _list(epiweeks),
        'locations': _list(locations)
      };
      return _request(callback, params);
    };

    Epidata.norostat = function(callback, auth, location, epiweeks) {
      var params;
      if (!((auth != null) && (location != null) && (epiweeks != null))) {
        throw {
          msg: '`auth`, `location`, and `epiweeks` are all required'
        };
      }
      params = {
        'source': 'norostat',
        'auth': auth,
        'location': location,
        'epiweeks': _list(epiweeks)
      };
      return _request(callback, params);
    };

    Epidata.meta_norostat = function(callback, auth) {
      var params;
      if (auth == null) {
        throw {
          msg: '`auth` is required'
        };
      }
      params = {
        'source': 'meta_norostat',
        'auth': auth
      };
      return _request(callback, params);
    };

    Epidata.afhsb = function(callback, auth, locations, epiweeks, flu_types) {
      var params;
      if (!((auth != null) && (locations != null) && (epiweeks != null) && (flu_types != null))) {
        throw {
          msg: '`auth`, `locations`, `epiweeks` and `flu_types` are all required'
        };
      }
      params = {
        'source': 'afhsb',
        'auth': auth,
        'locations': _list(locations),
        'epiweeks': _list(epiweeks),
        'flu_types': _list(flu_types)
      };
      return _request(callback, params);
    };

    Epidata.meta_afhsb = function(callback, auth) {
      var params;
      if (auth == null) {
        throw {
          msg: '`auth` is required'
        };
      }
      params = {
        'source': 'meta_afhsb',
        'auth': auth
      };
      return _request(callback, params);
    };

    Epidata.nidss_flu = function(callback, regions, epiweeks, issues, lag) {
      var params;
      if (!((regions != null) && (epiweeks != null))) {
        throw {
          msg: '`regions` and `epiweeks` are both required'
        };
      }
      if ((issues != null) && (lag != null)) {
        throw {
          msg: '`issues` and `lag` are mutually exclusive'
        };
      }
      params = {
        'source': 'nidss_flu',
        'regions': _list(regions),
        'epiweeks': _list(epiweeks)
      };
      if (issues != null) {
        params.issues = _list(issues);
      }
      if (lag != null) {
        params.lag = lag;
      }
      return _request(callback, params);
    };

    Epidata.nidss_dengue = function(callback, locations, epiweeks) {
      var params;
      if (!((locations != null) && (epiweeks != null))) {
        throw {
          msg: '`locations` and `epiweeks` are both required'
        };
      }
      params = {
        'source': 'nidss_dengue',
        'locations': _list(locations),
        'epiweeks': _list(epiweeks)
      };
      return _request(callback, params);
    };

    Epidata.delphi = function(callback, system, epiweek) {
      var params;
      if (!((system != null) && (epiweek != null))) {
        throw {
          msg: '`system` and `epiweek` are both required'
        };
      }
      params = {
        'source': 'delphi',
        'system': system,
        'epiweek': epiweek
      };
      return _request(callback, params);
    };

    Epidata.sensors = function(callback, auth, names, locations, epiweeks) {
      var params;
      if (!((auth != null) && (names != null) && (locations != null) && (epiweeks != null))) {
        throw {
          msg: '`auth`, `names`, `locations`, and `epiweeks` are all required'
        };
      }
      params = {
        'source': 'sensors',
        'auth': auth,
        'names': _list(names),
        'locations': _list(locations),
        'epiweeks': _list(epiweeks)
      };
      return _request(callback, params);
    };

    Epidata.nowcast = function(callback, locations, epiweeks) {
      var params;
      if (!((locations != null) && (epiweeks != null))) {
        throw {
          msg: '`locations` and `epiweeks` are both required'
        };
      }
      params = {
        'source': 'nowcast',
        'locations': _list(locations),
        'epiweeks': _list(epiweeks)
      };
      return _request(callback, params);
    };

    Epidata.meta = function(callback) {
      return _request(callback, {
        'source': 'meta'
      });
    };

    return Epidata;

  })();

  (typeof exports !== "undefined" && exports !== null ? exports : window).Epidata = Epidata;

}).call(this);
